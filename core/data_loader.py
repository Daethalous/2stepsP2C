import json
import os
from typing import Any, NamedTuple, List, Dict, Optional

from core.exceptions import PipelineError
from core.utils import (
    extract_planning,
    content_to_json,
    parse_structured_json,
    validate_required_keys,
)


def load_paper_content(paper_format: str,
                       pdf_json_path: str = None,
                       pdf_latex_path: str = None) -> Any:
    if paper_format == "JSON":
        with open(f'{pdf_json_path}') as f:
            return json.load(f)
    elif paper_format == "LaTeX":
        with open(f'{pdf_latex_path}') as f:
            return f.read()
    else:
        raise PipelineError(f"Invalid paper format '{paper_format}'. Please select either 'JSON' or 'LaTeX.")


class PipelineContext(NamedTuple):
    config_yaml: str
    context_lst: list
    task_list: dict
    todo_file_lst: list
    logic_analysis_dict: Dict[str, str]


def _get_key(d: dict, *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_pipeline_context(output_dir: str) -> PipelineContext:
    with open(f'{output_dir}/planning_config.yaml') as f:
        config_yaml = f.read()

    context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

    if os.path.exists(f'{output_dir}/task_list.json'):
        with open(f'{output_dir}/task_list.json') as f:
            task_list = json.load(f)
    else:
        task_list = parse_structured_json(context_lst[2])
        if not task_list:
            task_list = content_to_json(context_lst[2])

    if not validate_required_keys(task_list, ["Task list", "Logic Analysis"]):
        # Allow key variants and continue with fallback selection below.
        if not isinstance(task_list, dict):
            task_list = {}

    todo_file_lst = _get_key(task_list, 'Task list', 'task_list', 'task list', 'Task List')
    if todo_file_lst is None:
        raise PipelineError("'Task list' does not exist. Please re-generate the planning.")

    logic_analysis = _get_key(task_list, 'Logic Analysis', 'logic_analysis', 'logic analysis', 'Logic analysis')
    if logic_analysis is None:
        raise PipelineError("'Logic Analysis' does not exist. Please re-generate the planning.")

    logic_analysis_dict = {}
    for desc in logic_analysis:
        logic_analysis_dict[desc[0]] = desc[1]

    return PipelineContext(
        config_yaml=config_yaml,
        context_lst=context_lst,
        task_list=task_list,
        todo_file_lst=todo_file_lst,
        logic_analysis_dict=logic_analysis_dict,
    )
