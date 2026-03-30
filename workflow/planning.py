import json
from tqdm import tqdm
import argparse
import os
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content
from core.logger import get_logger
from core.prompts.templates import render_prompt, load_prompt
from core.utils import (
    print_response,
    print_log_cost,
    save_accumulated_cost,
    build_baseline_repo_context_compact,
    format_paper_content_for_prompt,
    parse_structured_json,
    validate_required_keys,
)

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def run_planning(paper_name: str, gpt_version: str, output_dir: str,
                 paper_format: str = "JSON",
                 pdf_json_path: str = None,
                 pdf_latex_path: str = None,
                 prompt_set: str = None,
                 baseline_repo_dir: str = None) -> None:

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=24000)

    baseline_repo_summary = ""
    baseline_repo_code = ""
    if prompt_set == "feature" and baseline_repo_dir:
        baseline_repo_summary, baseline_repo_code = build_baseline_repo_context_compact(
            baseline_repo_dir=baseline_repo_dir,
            max_files=120,
            max_chars_per_file=1800,
        )

    plan_msg = [
            {'role': "system", "content": render_prompt(
                _prompt_path(prompt_set, "planning_system.txt"),
                paper_format=paper_format)},
            {"role": "user", "content": render_prompt(
                _prompt_path(prompt_set, "planning_user_plan.txt"),
                paper_content=paper_content_prompt,
                baseline_repo_summary=baseline_repo_summary)}]

    file_list_msg = [
            {"role": "user", "content": render_prompt(
                _prompt_path(prompt_set, "planning_user_design.txt"),
                baseline_repo_code=baseline_repo_code)}]

    task_list_msg = [
            {'role': 'user', 'content': load_prompt(
                _prompt_path(prompt_set, "planning_user_task.txt"))}]

    config_msg = [
            {'role': 'user', 'content': load_prompt(
                _prompt_path(prompt_set, "planning_user_config.txt"))}]

    def api_call(msg, gpt_version):
        return chat_completion_with_retry(client, gpt_version, msg)

    responses = []
    trajectories = []
    total_accumulated_cost = 0
    feature_required_keys = {
        0: ["Plan Summary", "Method Requirements", "File-Level Strategy", "Infrastructure Boundaries", "Risk List", "Anything UNCLEAR"],
        1: ["Baseline architecture summary", "Paper algorithm summary", "Injection points", "New files needed", "Files unchanged", "Coverage check", "Anything UNCLEAR"],
        2: ["Logic Analysis", "Task list", "Anything UNCLEAR"],
    }

    for idx, instruction_msg in enumerate([plan_msg, file_list_msg, task_list_msg, config_msg]):
        current_stage = ""
        if idx == 0 :
            current_stage = f"[Planning] Overall plan"
        elif idx == 1:
            current_stage = f"[Planning] Architecture design"
        elif idx == 2:
            current_stage = f"[Planning] Logic design"
        elif idx == 3:
            current_stage = f"[Planning] Configuration file generation"
        logger.info(current_stage)

        trajectories.extend(instruction_msg)

        completion = None
        completion_json = None
        for attempt in range(3):
            completion = api_call(trajectories, gpt_version)
            completion_json = json.loads(completion.model_dump_json())
            if prompt_set != "feature" or idx == 3:
                break
            required = feature_required_keys.get(idx, [])
            model_text = completion_json["choices"][0]["message"]["content"]
            payload = parse_structured_json(model_text)
            if validate_required_keys(payload, required):
                break
            trajectories.append({
                "role": "assistant",
                "content": model_text,
            })
            trajectories.append({
                "role": "user",
                "content": (
                    "Your output format is invalid. Return ONLY [CONTENT]{json}[/CONTENT] "
                    f"with all required keys: {required}. Do not include any extra text."
                ),
            })
            logger.warning(f"[Planning] Output schema invalid at stage {idx}, retry {attempt+1}/3")

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/planning_response.json', 'w') as f:
        json.dump(responses, f)

    with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
        json.dump(trajectories, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--gpt_version', type=str)
    parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    run_planning(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
    )
