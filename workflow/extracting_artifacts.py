import json
import re
import os
import argparse
import shutil
import yaml
from core.logger import get_logger
from core.utils import extract_planning, content_to_json, format_json_data, extract_content_block

logger = get_logger(__name__)


def _strip_reasoning_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for marker in ("</think>", "</redacted_thinking>"):
        if marker in text:
            text = text.split(marker)[-1]
    return text.strip()


def _extract_yaml_content(text: str) -> str:
    text = _strip_reasoning_markers(text)
    if not text:
        return ""

    fenced = re.search(r"```yaml\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    escaped_fenced = re.search(r"```yaml\\n(.*?)\\n```", text, re.DOTALL)
    if escaped_fenced:
        return escaped_fenced.group(1).strip()

    payload = extract_content_block(text)
    if payload and payload != text:
        return payload.strip()

    return ""


def _load_planning_stage_outputs(output_dir: str) -> list:
    planning_response_path = os.path.join(output_dir, "planning_response.json")
    if os.path.exists(planning_response_path):
        try:
            with open(planning_response_path, encoding="utf-8") as f:
                responses = json.load(f)
            if isinstance(responses, list):
                context_lst = []
                for item in responses:
                    content = (
                        item.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    content = _strip_reasoning_markers(content)
                    if content:
                        context_lst.append(content)
                if context_lst:
                    return context_lst
        except Exception as exc:
            logger.warning(f"Failed to load planning_response.json, fallback to trajectories: {exc}")
    return extract_planning(f'{output_dir}/planning_trajectories.json')


def run_extracting_artifacts(output_dir: str) -> None:
    context_lst = _load_planning_stage_outputs(output_dir)

    yaml_raw_content = ""
    yaml_content = ""
    for candidate in reversed(context_lst):
        extracted = _extract_yaml_content(candidate)
        if extracted:
            yaml_raw_content = candidate
            yaml_content = extracted
            break

    if not yaml_content:
        with open(f'{output_dir}/planning_trajectories.json', encoding='utf8') as f:
            traj = json.load(f)
        for turn in reversed(traj):
            if turn.get('role') != 'assistant':
                continue
            candidate = turn.get('content', '')
            extracted = _extract_yaml_content(candidate)
            if extracted:
                yaml_raw_content = candidate
                yaml_content = extracted
                break
        if not yaml_content and len(traj) > 0:
            yaml_raw_content = traj[-1].get('content', '')
            yaml_content = _extract_yaml_content(yaml_raw_content)

    config_path = f'{output_dir}/planning_config.yaml'
    if yaml_content:
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML extracted from planning trajectories: {exc}") from exc
        if not isinstance(parsed_yaml, dict):
            raise ValueError("Extracted planning config is not a YAML mapping.")
        with open(config_path, 'w', encoding='utf8') as f:
            f.write(yaml_content)
    else:
        raise FileNotFoundError(
            "No YAML planning config found in planning_trajectories.json. "
            "Expected either ```yaml ... ``` or [CONTENT]...[/CONTENT] in the final assistant response."
        )

    # ---------------------------------------

    artifact_output_dir = f"{output_dir}/planning_artifacts"

    os.makedirs(artifact_output_dir, exist_ok=True)

    arch_design = content_to_json(context_lst[1])
    logic_design = content_to_json(context_lst[2])

    formatted_arch_design = format_json_data(arch_design)
    formatted_logic_design = format_json_data(logic_design)

    with open(f"{artifact_output_dir}/1.1_overall_plan.txt", "w", encoding="utf-8") as f:
        f.write(context_lst[0])

    with open(f"{artifact_output_dir}/1.2_arch_design.txt", "w", encoding="utf-8") as f:
        f.write(formatted_arch_design)

    with open(f"{artifact_output_dir}/1.3_logic_design.txt", "w", encoding="utf-8") as f:
        f.write(formatted_logic_design)

    shutil.copy(config_path, f"{artifact_output_dir}/1.4_config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    run_extracting_artifacts(output_dir=args.output_dir)
