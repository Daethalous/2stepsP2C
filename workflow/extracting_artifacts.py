import json
import re
import os
import argparse
import shutil
from core.logger import get_logger
from core.utils import extract_planning, content_to_json, format_json_data

logger = get_logger(__name__)


def run_extracting_artifacts(output_dir: str) -> None:

    with open(f'{output_dir}/planning_trajectories.json', encoding='utf8') as f:
        traj = json.load(f)

    yaml_raw_content = ""
    # In the new optimized format, trajectories only contains assistant responses.
    # The config yaml should be the very last assistant response.
    for turn in reversed(traj):
        if turn.get('role') == 'assistant' and '```yaml' in turn.get('content', ''):
            yaml_raw_content = turn['content']
            break

    if not yaml_raw_content and len(traj) > 0:
        yaml_raw_content = traj[-1]['content']

    if "</think>" in yaml_raw_content:
        yaml_raw_content = yaml_raw_content.split("</think>")[-1]

    match = re.search(r"```yaml\n(.*?)\n```", yaml_raw_content, re.DOTALL)
    if match:
        yaml_content = match.group(1)
        with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
            f.write(yaml_content)
    else:
        match2 = re.search(r"```yaml\\n(.*?)\\n```", yaml_raw_content, re.DOTALL)
        if match2:
            yaml_content = match2.group(1)
            with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
                f.write(yaml_content)
        else:
            logger.warning("No YAML content found.")

    # ---------------------------------------

    artifact_output_dir = f"{output_dir}/planning_artifacts"

    os.makedirs(artifact_output_dir, exist_ok=True)

    context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

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

    shutil.copy(f"{output_dir}/planning_config.yaml", f"{artifact_output_dir}/1.4_config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    run_extracting_artifacts(output_dir=args.output_dir)
