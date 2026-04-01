import json
import os
import time
from tqdm import tqdm
import re
import copy
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
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


def _load_global_api_contract_stub(output_dir: str) -> str:
    contract_path = os.path.join(output_dir, "api_predefine_contract.pyi")
    if not os.path.exists(contract_path):
        return "(none)"
    try:
        with open(contract_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text if text else "(none)"
    except Exception:
        return "(none)"


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
    global_api_contract_stub = _load_global_api_contract_stub(output_dir)
    todo_file_lst = ctx.todo_file_lst
    done_file_lst = ['config.yaml']
    done_file_dict = {}
    new_definitions = []

    def _resolve_todo_candidates(todo_file_name: str):
        candidates = [todo_file_name]
        clean_name = sanitize_todo_file_name(todo_file_name)
        if clean_name and clean_name not in candidates:
            candidates.append(clean_name)
        return candidates

    def _is_entry_point_file(path_name: str) -> bool:
        lowered = path_name.lower()
        return any(tag in lowered for tag in ("main.py", "__init__.py", "factory", "registry", "runner", "train"))

    def _extract_new_definitions(code_text: str):
        matches = re.findall(r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", code_text, flags=re.MULTILINE)
        return matches[:20]

    def _find_analysis_response_path(todo_file_name: str):
        for candidate in _resolve_todo_candidates(todo_file_name):
            save_name = candidate.replace("/", "_")
            path = os.path.join(output_dir, f"{save_name}_simple_analysis_response.json")
            if os.path.exists(path):
                return path
        return None

    def _load_feature_file_code(todo_file_name: str):
        for candidate in _resolve_todo_candidates(todo_file_name):
            if live_repo_dir:
                live_path = os.path.join(live_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live"
            if baseline_repo_dir:
                baseline_path = os.path.join(baseline_repo_dir, candidate)
                if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as bf:
                        return bf.read(), "baseline"
        return None, "new"

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
            file_code, code_source = _load_feature_file_code(todo_file_name)
            if file_code is None:
                extra_kwargs["baseline_file_code"] = "(new file — no baseline code)"
                existing_files = sorted(list(done_file_dict.keys()))
                extra_kwargs["existing_files_summary"] = "\n".join(f"- {x}" for x in existing_files[:200]) if existing_files else "(none)"
            else:
                sig = extract_interface_signatures(file_code, max_lines=120)
                compact_code = file_code[:16000]
                if len(file_code) > 16000:
                    compact_code += "\n...(truncated for token budget)..."
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {code_source}\n"
                    "### Interface Signatures\n"
                    f"{sig}\n\n"
                    "### Code Snapshot\n"
                    f"{compact_code}"
                )
            if _is_entry_point_file(todo_file_name) and new_definitions:
                extra_kwargs["integration_hint"] = (
                    "You have created/updated these definitions in earlier files:\n"
                    + "\n".join(f"- {x}" for x in new_definitions[:50])
                    + "\nEnsure this entry/factory/registry file integrates them where required."
                )
            else:
                extra_kwargs["integration_hint"] = ""

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
            global_api_contract_stub=global_api_contract_stub,
            **extra_kwargs)}]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    detailed_logic_analysis_dict = {}
    for todo_file_name in todo_file_lst:
        if todo_file_name == "config.yaml":
            continue

        analysis_path = _find_analysis_response_path(todo_file_name)
        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        if analysis_path and os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                detailed_logic_analysis_response = json.load(f)
            detailed_logic_analysis_dict[clean_todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
        else:
            logger.warning(f"[WARNING] Analysis file not found for {todo_file_name}. Continue with empty analysis.")
            detailed_logic_analysis_dict[clean_todo_file_name] = ""

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

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        if len(clean_todo_file_name.strip()) == 0:
            logger.warning(f"[CODING] Skip empty/invalid todo file name: {todo_file_name}")
            continue

        save_todo_file_name_skip = clean_todo_file_name.replace("/", "_")
        skip_path = f'{artifact_output_dir}/{save_todo_file_name_skip}_coding.txt'
        if os.path.exists(skip_path):
            repo_file_path = os.path.join(output_repo_dir, clean_todo_file_name)
            if os.path.exists(repo_file_path):
                with open(repo_file_path, 'r') as f:
                    done_file_dict[clean_todo_file_name] = f.read()
            done_file_lst.append(clean_todo_file_name)
            logger.info(f"  [SKIP] artifact already exists: {skip_path}")
            continue

        instruction_msg = get_write_msg(
            clean_todo_file_name,
            detailed_logic_analysis_dict.get(clean_todo_file_name, ""),
            done_file_lst,
        )
        trajectories.extend(instruction_msg)
        if todo_idx == 0:
            _debug_log(
                run_id="initial",
                hypothesis_id="H2",
                location="workflow/coding.py:before_api_call",
                message="first trajectories shape",
                data={
                    "todo_file_name": clean_todo_file_name,
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
                        "todo_file_name": clean_todo_file_name,
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
            logger.warning(f"[CODING] Placeholder detected in {clean_todo_file_name}, retry {attempt+1}/3")
        
        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

        done_file_lst.append(clean_todo_file_name)

        os.makedirs(f'{output_repo_dir}', exist_ok=True)
        save_todo_file_name = clean_todo_file_name.replace("/", "_")

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w') as f:
            f.write(completion_json['choices'][0]['message']['content'])

        code = extract_code_from_content(message.content)
        if len(code) == 0:
            code = message.content 

        done_file_dict[clean_todo_file_name] = code
        new_definitions.extend(_extract_new_definitions(code))
        file_path = os.path.join(output_repo_dir, clean_todo_file_name)
        if os.path.isdir(file_path):
            logger.warning(f"[CODING] Skip writing because target path is a directory: {file_path}")
            continue
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
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
