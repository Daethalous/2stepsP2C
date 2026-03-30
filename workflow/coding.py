import json
import os
import time
from tqdm import tqdm
import re
import copy
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, load_pipeline_context
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.utils import (extract_code_from_content, print_response, print_log_cost,
                        load_accumulated_cost, save_accumulated_cost,
                        parse_feature_design, read_python_files,
                        format_paper_content_for_prompt,
                        extract_interface_signatures,
                        build_code_interface_summary,
                        contains_forbidden_placeholders)
import argparse

logger = get_logger(__name__)
DEBUG_LOG_PATH = "debug-084f81.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "084f81",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # #endregion


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def run_coding(paper_name: str, gpt_version: str, output_dir: str,
               output_repo_dir: str,
               paper_format: str = "JSON",
               pdf_json_path: str = None,
               pdf_latex_path: str = None,
               prompt_set: str = None,
               baseline_repo_dir: str = None,
               live_repo_dir: str = None) -> None:
    _debug_log(
        run_id="initial",
        hypothesis_id="H1",
        location="workflow/coding.py:run_coding.entry",
        message="run_coding entry",
        data={
            "file": __file__,
            "gpt_version": gpt_version,
            "output_dir": output_dir,
            "output_repo_dir": output_repo_dir,
            "prompt_set": prompt_set,
            "baseline_repo_dir": baseline_repo_dir,
            "live_repo_dir": live_repo_dir,
        },
    )

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=22000)
    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    context_lst = ctx.context_lst
    overview_prompt = format_paper_content_for_prompt(context_lst[0], max_chars=12000) if len(context_lst) > 0 else ""
    design_prompt = format_paper_content_for_prompt(context_lst[1], max_chars=16000) if len(context_lst) > 1 else ""
    task_prompt = format_paper_content_for_prompt(context_lst[2], max_chars=16000) if len(context_lst) > 2 else ""
    todo_file_lst = ctx.todo_file_lst
    done_file_lst = ['config.yaml']
    done_file_dict = {}

    if prompt_set == "feature":
        existing = read_python_files(output_repo_dir)
        for fname, content in existing.items():
            if fname not in done_file_dict:
                done_file_dict[fname] = content

    code_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "coding_system.txt"),
            paper_format=paper_format)}]

    def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst): 
        code_files = build_code_interface_summary(done_file_dict, done_file_lst, max_total_chars=14000)

        extra_kwargs = {}
        if prompt_set == "feature" and baseline_repo_dir:
            file_code = None
            if live_repo_dir:
                live_path = os.path.join(live_repo_dir, todo_file_name)
                if os.path.exists(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        file_code = lf.read()
            if file_code is None:
                baseline_file_path = os.path.join(baseline_repo_dir, todo_file_name)
                if os.path.exists(baseline_file_path):
                    with open(baseline_file_path, "r", encoding="utf-8") as bf:
                        file_code = bf.read()
            if file_code is None:
                extra_kwargs["baseline_file_code"] = "(new file — no baseline code)"
            else:
                sig = extract_interface_signatures(file_code, max_lines=120)
                compact_code = file_code[:16000]
                if len(file_code) > 16000:
                    compact_code += "\n...(truncated for token budget)..."
                extra_kwargs["baseline_file_code"] = (
                    "### Interface Signatures\n"
                    f"{sig}\n\n"
                    "### Code Snapshot\n"
                    f"{compact_code}"
                )

        write_msg=[
{'role': 'user', "content": render_prompt(
            _prompt_path(prompt_set, "coding_user.txt"),
            paper_content=paper_content_prompt,
            overview=overview_prompt,
            design=design_prompt,
            task=task_prompt,
            config_yaml=config_yaml,
            code_files=code_files,
            todo_file_name=todo_file_name,
            done_file_lst=done_file_lst,
            detailed_logic_analysis=detailed_logic_analysis,
            **extra_kwargs)}]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    detailed_logic_analysis_dict = {}
    for todo_file_name in todo_file_lst:
        save_todo_file_name = todo_file_name.replace("/", "_")

        if todo_file_name == "config.yaml":
            continue
        
        with open(f"{output_dir}/{save_todo_file_name}_simple_analysis_response.json") as f:
            detailed_logic_analysis_response = json.load(f)
        detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']

    artifact_output_dir = f'{output_dir}/coding_artifacts'
    os.makedirs(artifact_output_dir, exist_ok=True)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
        responses = []
        trajectories = copy.deepcopy(code_msg)

        current_stage = f"[CODING] {todo_file_name}"
        logger.info(current_stage)

        if todo_file_name == "config.yaml":
            continue

        save_todo_file_name_skip = todo_file_name.replace("/", "_")
        skip_path = f'{artifact_output_dir}/{save_todo_file_name_skip}_coding.txt'
        if os.path.exists(skip_path):
            repo_file_path = f"{output_repo_dir}/{todo_file_name}"
            if os.path.exists(repo_file_path):
                with open(repo_file_path, 'r') as f:
                    done_file_dict[todo_file_name] = f.read()
            done_file_lst.append(todo_file_name)
            logger.info(f"  [SKIP] artifact already exists: {skip_path}")
            continue

        instruction_msg = get_write_msg(todo_file_name, detailed_logic_analysis_dict[todo_file_name], done_file_lst)
        trajectories.extend(instruction_msg)
        if todo_idx == 0:
            _debug_log(
                run_id="initial",
                hypothesis_id="H2",
                location="workflow/coding.py:before_api_call",
                message="first trajectories shape",
                data={
                    "todo_file_name": todo_file_name,
                    "msg_count": len(trajectories),
                    "roles": [m.get("role") for m in trajectories],
                    "content_types": [type(m.get("content")).__name__ for m in trajectories],
                },
            )
            try:
                dumped = json.dumps(trajectories, ensure_ascii=False)
                _debug_log(
                    run_id="initial",
                    hypothesis_id="H3",
                    location="workflow/coding.py:before_api_call",
                    message="trajectories json dump ok",
                    data={"dump_len": len(dumped)},
                )
            except Exception as dump_exc:
                _debug_log(
                    run_id="initial",
                    hypothesis_id="H3",
                    location="workflow/coding.py:before_api_call",
                    message="trajectories json dump failed",
                    data={"error_type": type(dump_exc).__name__, "error": str(dump_exc)},
                )

        completion = None
        for attempt in range(3):
            try:
                completion = api_call(trajectories)
            except Exception as api_exc:
                _debug_log(
                    run_id="initial",
                    hypothesis_id="H4",
                    location="workflow/coding.py:api_call",
                    message="api call failed",
                    data={
                        "todo_file_name": todo_file_name,
                        "error_type": type(api_exc).__name__,
                        "error": str(api_exc),
                        "status_code": getattr(api_exc, "status_code", None),
                    },
                )
                raise
            completion_json_try = json.loads(completion.model_dump_json())
            content_try = completion_json_try["choices"][0]["message"]["content"]
            code_try = extract_code_from_content(content_try) or content_try
            if not contains_forbidden_placeholders(code_try):
                break
            trajectories.append({"role": "assistant", "content": content_try})
            trajectories.append({
                "role": "user",
                "content": (
                    "Output rejected: code still contains NotImplementedError/TODO/placeholder. "
                    "Return ONLY the complete corrected file code, no explanations."
                ),
            })
            logger.warning(f"[CODING] Placeholder detected in {todo_file_name}, retry {attempt+1}/3")
        
        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

        done_file_lst.append(todo_file_name)

        os.makedirs(f'{output_repo_dir}', exist_ok=True)
        save_todo_file_name = todo_file_name.replace("/", "_")

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w') as f:
            f.write(completion_json['choices'][0]['message']['content'])

        code = extract_code_from_content(message.content)
        if len(code) == 0:
            code = message.content 

        done_file_dict[todo_file_name] = code
        if save_todo_file_name != todo_file_name:
            todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
            os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)

        with open(f"{output_repo_dir}/{todo_file_name}", 'w') as f:
            f.write(code)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--gpt_version', type=str, default="o3-mini")
    parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--output_repo_dir', type=str, default="")
    args = parser.parse_args()

    run_coding(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
    )
