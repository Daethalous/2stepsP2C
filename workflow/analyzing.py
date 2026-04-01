import json
import os
from tqdm import tqdm
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.utils import (print_response, print_log_cost, load_accumulated_cost,
                        save_accumulated_cost, parse_feature_design,
                        get_injection_info_for_file,
                        format_paper_content_for_prompt,
                        extract_interface_signatures,
                        parse_structured_json,
                        validate_required_keys)
import copy
import argparse

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def run_analyzing(paper_name: str, gpt_version: str, output_dir: str,
                  paper_format: str = "JSON",
                  pdf_json_path: str = None,
                  pdf_latex_path: str = None,
                  prompt_set: str = None,
                  baseline_repo_dir: str = None,
                  live_repo_dir: str = None) -> None:

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
    logic_analysis_dict = dict(ctx.logic_analysis_dict)

    injection_points = []
    if prompt_set == "feature":
        injection_points, _, _ = parse_feature_design(context_lst[1])

    done_file_lst = ['config.yaml']

    analysis_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_system.txt"),
            paper_format=paper_format)}]

    def _load_feature_file_code(todo_file_name: str):
        candidates = [todo_file_name]
        clean_name = sanitize_todo_file_name(todo_file_name)
        if clean_name and clean_name not in candidates:
            candidates.append(clean_name)
        if prompt_set == "feature" and live_repo_dir:
            for candidate in candidates:
                live_path = os.path.join(live_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live"
        if prompt_set == "feature" and baseline_repo_dir:
            for candidate in candidates:
                baseline_path = os.path.join(baseline_repo_dir, candidate)
                if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as bf:
                        return bf.read(), "baseline"
        return None, "new"

    def get_write_msg(todo_file_name, todo_file_desc):
        draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
        if len(todo_file_desc.strip()) == 0:
            draft_desc = f"Write the logic analysis in '{todo_file_name}'."

        extra_kwargs = {}
        if prompt_set == "feature" and baseline_repo_dir:
            file_code, code_source = _load_feature_file_code(todo_file_name)
            if file_code is None:
                extra_kwargs["baseline_file_code"] = "(new file — no baseline/live code)"
            else:
                sig = extract_interface_signatures(file_code, max_lines=120)
                compact_code = file_code[:14000]
                if len(file_code) > 14000:
                    compact_code += "\n...(truncated for token budget)..."
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {code_source}\n"
                    "### Interface Signatures\n"
                    f"{sig}\n\n"
                    "### Code Snapshot\n"
                    f"{compact_code}"
                )
            extra_kwargs["injection_info"] = get_injection_info_for_file(
                injection_points, todo_file_name)

        write_msg=[{'role': 'user', "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_user.txt"),
            paper_content=paper_content_prompt,
            overview=overview_prompt,
            design=design_prompt,
            task=task_prompt,
            config_yaml=config_yaml,
            draft_desc=draft_desc,
            todo_file_name=todo_file_name,
            **extra_kwargs)}]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    artifact_output_dir = f'{output_dir}/analyzing_artifacts'
    os.makedirs(artifact_output_dir, exist_ok=True)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    for todo_file_name in tqdm(todo_file_lst):
        responses = []
        trajectories = copy.deepcopy(analysis_msg)

        current_stage = f"[ANALYSIS] {todo_file_name}"
        logger.info(current_stage)
        if todo_file_name == "config.yaml":
            continue

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        _skip_name = clean_todo_file_name.replace("/", "_")
        _skip_path = os.path.join(artifact_output_dir, f"{_skip_name}_simple_analysis.txt")
        if os.path.exists(_skip_path):
            logger.info(f"  [SKIP] artifact already exists: {_skip_path}")
            done_file_lst.append(clean_todo_file_name)
            continue

        if clean_todo_file_name not in logic_analysis_dict:
            logic_analysis_dict[clean_todo_file_name] = ""
            
        instruction_msg = get_write_msg(clean_todo_file_name, logic_analysis_dict[clean_todo_file_name])
        trajectories.extend(instruction_msg)

        completion = None
        completion_json = None
        for attempt in range(3):
            completion = api_call(trajectories)
            completion_json = json.loads(completion.model_dump_json())
            if prompt_set != "feature":
                break
            model_text = completion_json["choices"][0]["message"]["content"]
            payload = parse_structured_json(model_text)
            required = [
                "file", "modification_steps", "interface_contract_checklist",
                "config_keys_used", "test_focus", "blocked_items",
            ]
            if validate_required_keys(payload, required):
                break
            trajectories.append({"role": "assistant", "content": model_text})
            trajectories.append({
                "role": "user",
                "content": (
                    "Your output format is invalid. Return ONLY [CONTENT]{json}[/CONTENT] "
                    f"with required keys: {required}. No extra text."
                ),
            })
            logger.warning(f"[ANALYSIS] Schema invalid for {clean_todo_file_name}, retry {attempt+1}/3")
        responses.append(completion_json)
        
        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        save_todo_file_name = clean_todo_file_name.replace("/", "_")
        artifact_file_path = os.path.join(artifact_output_dir, f"{save_todo_file_name}_simple_analysis.txt")
        os.makedirs(os.path.dirname(artifact_file_path), exist_ok=True)
        with open(artifact_file_path, 'w') as f:
            f.write(completion_json['choices'][0]['message']['content'])

        done_file_lst.append(clean_todo_file_name)

        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_response.json"), 'w') as f:
            json.dump(responses, f)

        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_trajectories.json"), 'w') as f:
            json.dump(trajectories, f)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--gpt_version', type=str, default="o3-mini")
    parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    run_analyzing(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
    )
