import json
import os
import re
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


def sanitize_todo_file_name(name: str) -> str:
    """Normalize task-list file entries into stable relative file paths."""
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().strip("'\"")
    # Drop common list prefixes: "1. ", "1) ", "- ", "* "
    cleaned = re.sub(r"^\s*(?:[-*]\s+|\d+\s*[.)]\s+)", "", cleaned)
    # Keep only path-like head when the item contains descriptions.
    # Examples:
    #   "src/a.py - add foo" -> "src/a.py"
    #   "main.py: update training loop" -> "main.py"
    m = re.match(r"^([A-Za-z0-9_./\\-]+\.(?:py|yaml|yml|json))\b", cleaned)
    if m:
        cleaned = m.group(1)
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/{2,}", "/", cleaned).strip().lstrip("./")
    return cleaned


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

    logic_analysis = _get_key(task_list, 'Logic Analysis', 'logic_analysis', 'logic analysis', 'Logic analysis')
    if logic_analysis is None:
        raise PipelineError("'Logic Analysis' does not exist. Please re-generate the planning.")

    if todo_file_lst is None:
        # Fallback for imperfect planning outputs: derive todo files from logic-analysis entries.
        todo_file_lst = []
        for desc in logic_analysis:
            if isinstance(desc, (list, tuple)) and len(desc) > 0:
                candidate = sanitize_todo_file_name(str(desc[0]))
                if candidate:
                    todo_file_lst.append(candidate)
        if len(todo_file_lst) == 0:
            raise PipelineError("'Task list' does not exist and no fallback could be extracted from 'Logic Analysis'.")

    normalized_todo = []
    seen_todo = set()
    for item in todo_file_lst:
        clean_item = sanitize_todo_file_name(str(item))
        if not clean_item or clean_item in seen_todo:
            continue
        normalized_todo.append(clean_item)
        seen_todo.add(clean_item)
    todo_file_lst = normalized_todo

    logic_analysis_dict = {}
    for desc in logic_analysis:
        if not isinstance(desc, (list, tuple)) or len(desc) < 2:
            continue
        file_key = sanitize_todo_file_name(str(desc[0]))
        if not file_key:
            continue
        logic_analysis_dict[file_key] = desc[1]

    return PipelineContext(
        config_yaml=config_yaml,
        context_lst=context_lst,
        task_list=task_list,
        todo_file_lst=todo_file_lst,
        logic_analysis_dict=logic_analysis_dict,
    )
