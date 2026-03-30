import argparse
import json
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.data_loader import load_paper_content
from core.llm_engine import create_client, chat_completion_with_retry
from core.parser.pdf_process import run_pdf_process
from core.prompts.templates import render_prompt, load_prompt
from core.utils import format_paper_content_for_prompt


def _read_cached_assistant_content(response_path: str) -> str:
    with open(response_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["choices"][0]["message"]["content"]


def run_baseline_plan_and_design(
    input_json_path: str,
    output_dir: str,
    gpt_version: str,
    force_preprocess: bool = False,
    force_overall: bool = False,
    force_design: bool = False,
    force_task: bool = False,
    force_config: bool = False,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    os.makedirs(output_dir, exist_ok=True)

    cleaned_json_path = os.path.join(output_dir, "iTransformer_cleaned.json")
    if force_preprocess or not os.path.exists(cleaned_json_path):
        run_pdf_process(input_json_path, cleaned_json_path)

    paper_content = load_paper_content(
        paper_format="JSON",
        pdf_json_path=cleaned_json_path,
        pdf_latex_path=None,
    )
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=24000)

    overall_messages = [
        {
            "role": "system",
            "content": render_prompt("baseline/planning_system.txt", paper_format="JSON"),
        },
        {
            "role": "user",
            "content": render_prompt(
                "baseline/planning_user_plan.txt",
                paper_content=paper_content_prompt,
                baseline_repo_summary="",
            ),
        },
    ]

    client = create_client()
    overall_messages_path = os.path.join(output_dir, "planning_overall_plan_messages.json")
    overall_response_path = os.path.join(output_dir, "planning_overall_plan_response.json")

    if force_overall or not os.path.exists(overall_response_path):
        completion = chat_completion_with_retry(client, gpt_version, overall_messages)
        completion_json = json.loads(completion.model_dump_json())
        with open(overall_messages_path, "w", encoding="utf-8") as f:
            json.dump(overall_messages, f, ensure_ascii=False, indent=2)
        with open(overall_response_path, "w", encoding="utf-8") as f:
            json.dump(completion_json, f, ensure_ascii=False, indent=2)
        overall_assistant_content = completion_json["choices"][0]["message"]["content"]
    else:
        overall_assistant_content = _read_cached_assistant_content(overall_response_path)

    design_messages = [
        overall_messages[0],
        overall_messages[1],
        {"role": "assistant", "content": overall_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_design.txt")},
    ]
    design_messages_path = os.path.join(output_dir, "planning_arch_design_messages.json")
    design_response_path = os.path.join(output_dir, "planning_arch_design_response.json")

    if force_design or not os.path.exists(design_response_path):
        design_completion = chat_completion_with_retry(client, gpt_version, design_messages)
        design_completion_json = json.loads(design_completion.model_dump_json())
        with open(design_messages_path, "w", encoding="utf-8") as f:
            json.dump(design_messages, f, ensure_ascii=False, indent=2)
        with open(design_response_path, "w", encoding="utf-8") as f:
            json.dump(design_completion_json, f, ensure_ascii=False, indent=2)
        design_assistant_content = design_completion_json["choices"][0]["message"]["content"]
    else:
        design_assistant_content = _read_cached_assistant_content(design_response_path)

    task_messages = [
        overall_messages[0],
        overall_messages[1],
        {"role": "assistant", "content": overall_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_design.txt")},
        {"role": "assistant", "content": design_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_task.txt")},
    ]
    task_messages_path = os.path.join(output_dir, "planning_logic_task_messages.json")
    task_response_path = os.path.join(output_dir, "planning_logic_task_response.json")

    if force_task or not os.path.exists(task_response_path):
        task_completion = chat_completion_with_retry(client, gpt_version, task_messages)
        task_completion_json = json.loads(task_completion.model_dump_json())
        with open(task_messages_path, "w", encoding="utf-8") as f:
            json.dump(task_messages, f, ensure_ascii=False, indent=2)
        with open(task_response_path, "w", encoding="utf-8") as f:
            json.dump(task_completion_json, f, ensure_ascii=False, indent=2)
        task_assistant_content = task_completion_json["choices"][0]["message"]["content"]
    else:
        task_assistant_content = _read_cached_assistant_content(task_response_path)

    config_messages = [
        overall_messages[0],
        overall_messages[1],
        {"role": "assistant", "content": overall_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_design.txt")},
        {"role": "assistant", "content": design_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_task.txt")},
        {"role": "assistant", "content": task_assistant_content},
        {"role": "user", "content": load_prompt("baseline/planning_user_config.txt")},
    ]
    config_messages_path = os.path.join(output_dir, "planning_config_messages.json")
    config_response_path = os.path.join(output_dir, "planning_config_response.json")

    if force_config or not os.path.exists(config_response_path):
        config_completion = chat_completion_with_retry(client, gpt_version, config_messages)
        config_completion_json = json.loads(config_completion.model_dump_json())
        with open(config_messages_path, "w", encoding="utf-8") as f:
            json.dump(config_messages, f, ensure_ascii=False, indent=2)
        with open(config_response_path, "w", encoding="utf-8") as f:
            json.dump(config_completion_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] cleaned json: {cleaned_json_path}")
    print(f"[DONE] overall plan messages: {overall_messages_path}")
    print(f"[DONE] overall plan response: {overall_response_path}")
    print(f"[DONE] arch design messages: {design_messages_path}")
    print(f"[DONE] arch design response: {design_response_path}")
    print(f"[DONE] logic task messages: {task_messages_path}")
    print(f"[DONE] logic task response: {task_response_path}")
    print(f"[DONE] config messages: {config_messages_path}")
    print(f"[DONE] config response: {config_response_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path",
        type=str,
        default=os.path.join("data", "paper2code", "paper2code_data", "iclr2024", "iTransformer.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_planning_overall_only"),
    )
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_overall", action="store_true")
    parser.add_argument("--force_design", action="store_true")
    parser.add_argument("--force_task", action="store_true")
    parser.add_argument("--force_config", action="store_true")
    args = parser.parse_args()

    run_baseline_plan_and_design(
        input_json_path=args.input_json_path,
        output_dir=args.output_dir,
        gpt_version=args.gpt_version,
        force_preprocess=args.force_preprocess,
        force_overall=args.force_overall,
        force_design=args.force_design,
        force_task=args.force_task,
        force_config=args.force_config,
    )
