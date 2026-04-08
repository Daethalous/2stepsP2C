import json
import argparse
import os
import re
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, _build_feature_metadata
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.repo_index import (
    build_modification_closure,
    build_repo_index,
    save_repo_index,
    summarize_entrypoint_index,
    summarize_repo_index,
)
from core.utils import (
    print_response,
    print_log_cost,
    save_accumulated_cost,
    format_paper_content_for_prompt,
    content_to_json,
    parse_structured_json,
    validate_required_keys,
)

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def _get_repo_content(repo_dir: str) -> str:
    if not repo_dir or not os.path.isdir(repo_dir):
        return "(none)"

    parts = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file_name in sorted(files):
            lower_name = file_name.lower()
            if not lower_name.endswith((".py", ".yaml", ".yml", ".json")):
                continue
            if lower_name.startswith(("test", "setup")):
                continue
            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path, repo_dir).replace("\\", "/")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as exc:
                logger.warning(f"[Planning] skip repo file {file_path}: {exc}")
                continue
            if rel_path.endswith(".py"):
                fence_lang = "python"
            elif rel_path.endswith((".yaml", ".yml")):
                fence_lang = "yaml"
            else:
                fence_lang = "json"
            parts.append(f"\n\n## File: {rel_path}\n```{fence_lang}\n{content}\n```")

    if not parts:
        return "(none)"
    return "".join(parts)


def _parse_planning_payload(model_text: str) -> dict:
    payload = parse_structured_json(model_text)
    if payload:
        return payload
    payload = content_to_json(model_text)
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_relative_task_path(value: str) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip("'\"")
    cleaned = re.sub(r"^\s*(?:[-*]\s+|\d+\s*[.)]\s+)", "", cleaned)
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/{2,}", "/", cleaned).strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _has_generic_file_extension(candidate: str) -> bool:
    base_name = candidate.rsplit("/", 1)[-1]
    if "." not in base_name:
        return False
    stem, suffix = base_name.rsplit(".", 1)
    return bool(stem and suffix and re.fullmatch(r"[A-Za-z0-9._-]+", suffix))


def _is_valid_relative_task_path(path_text: str) -> bool:
    if not isinstance(path_text, str):
        return False
    candidate = _normalize_relative_task_path(path_text)
    if not candidate:
        return False
    if os.path.isabs(candidate):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return False
    if any(token in candidate for token in ("..", "*", "?", "[", "]")):
        return False
    if any(token in candidate for token in (":", ",", ";")):
        return False
    if any(ch.isspace() for ch in candidate):
        return False
    if candidate.endswith("/"):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", candidate):
        return False
    path_parts = candidate.split("/")
    if any(part in ("", ".", "..") for part in path_parts):
        return False
    if not all(re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in path_parts):
        return False
    if not _has_generic_file_extension(candidate):
        return False
    return True


def _normalize_logic_analysis_paths(logic_analysis: object) -> set[str]:
    normalized = set()
    if not isinstance(logic_analysis, list):
        return normalized
    for item in logic_analysis:
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            continue
        candidate = _normalize_relative_task_path(str(item[0]))
        if _is_valid_relative_task_path(candidate):
            normalized.add(candidate)
    return normalized


def _validate_logic_analysis_entries(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Logic Analysis payload is not a dict."
    logic_analysis = payload.get("Logic Analysis")
    if not isinstance(logic_analysis, list) or not logic_analysis:
        return False, "Logic Analysis must be a non-empty list."
    seen = set()
    logic_paths = []
    for entry in logic_analysis:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return False, "Each Logic Analysis item must be [relative_file_path, description]."
        file_key = _normalize_relative_task_path(str(entry[0]))
        if not _is_valid_relative_task_path(file_key):
            return False, (
                f"Logic Analysis file key '{entry[0]}' is invalid. "
                "Use a single relative file path only."
            )
        if file_key in seen:
            return False, f"Logic Analysis contains duplicate file key '{file_key}'."
        seen.add(file_key)
        logic_paths.append(file_key)
    task_list = payload.get("Task list")
    if isinstance(task_list, list) and task_list:
        normalized_items = []
        for item in task_list:
            if not isinstance(item, str):
                return False, "Task list items must be strings."
            normalized = _normalize_relative_task_path(item)
            if not _is_valid_relative_task_path(normalized):
                return False, f"Task list item '{item}' is invalid."
            normalized_items.append(normalized)
        if set(normalized_items) != set(logic_paths):
            return False, "Task list and Logic Analysis must contain the same file set."
    return True, ""


def _validate_task_payload_paths(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    task_list = payload.get("Task list")
    if not isinstance(task_list, list) or not task_list:
        return False
    normalized_items = []
    seen = set()
    for item in task_list:
        if not isinstance(item, str):
            return False
        normalized = _normalize_relative_task_path(item)
        if not _is_valid_relative_task_path(normalized):
            return False
        if normalized in seen:
            return False
        seen.add(normalized)
        normalized_items.append(normalized)
    logic_ok, _ = _validate_logic_analysis_entries(payload)
    if not logic_ok:
        return False
    logic_paths = _normalize_logic_analysis_paths(payload.get("Logic Analysis"))
    if set(normalized_items) != logic_paths:
        return False
    return True


def _normalize_task_payload_for_snapshot(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    normalized_payload = dict(payload)
    task_list = normalized_payload.get("Task list")
    if isinstance(task_list, list):
        normalized_payload["Task list"] = [
            _normalize_relative_task_path(item)
            for item in task_list
            if _is_valid_relative_task_path(_normalize_relative_task_path(item))
        ]
    logic_analysis = normalized_payload.get("Logic Analysis")
    if isinstance(logic_analysis, list):
        normalized_logic = []
        for entry in logic_analysis:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            file_path = _normalize_relative_task_path(str(entry[0]))
            if not _is_valid_relative_task_path(file_path):
                continue
            normalized_logic.append([file_path, *list(entry[1:])])
        normalized_payload["Logic Analysis"] = normalized_logic
    return normalized_payload


def _get_required_keys(prompt_set: str, idx: int) -> list:
    baseline_required_keys = {
        1: [
            "Implementation approach",
            "File list",
            "Data structures and interfaces",
            "Program call flow",
            "Anything UNCLEAR",
        ],
        2: [
            "Required packages",
            "Required Other language third-party packages",
            "Logic Analysis",
            "Task list",
            "Full API spec",
            "Shared Knowledge",
            "Anything UNCLEAR",
        ],
    }
    feature_required_keys = {
        1: [
            "Implementation approach",
            "File list",
            "Primary Entry Points",
            "Execution Chain",
            "Core Replacement Targets",
            "New Files Justification",
            "Registry/Factory Touchpoints",
            "File Naming Review",
            "Data structures and interfaces",
            "Program call flow",
            "Anything UNCLEAR",
        ],
        2: [
            "Required packages",
            "Required Other language third-party packages",
            "Logic Analysis",
            "Task list",
            "Modification Closure",
            "Callsite Update List",
            "Public Interface Changes",
            "Forbidden File Names",
            "Full API spec",
            "Shared Knowledge",
            "Anything UNCLEAR",
        ],
    }
    required_map = feature_required_keys if prompt_set == "feature" else baseline_required_keys
    return required_map.get(idx, [])


def _write_task_list_snapshot(output_dir: str, task_payload: dict) -> None:
    if not isinstance(task_payload, dict):
        return
    task_payload = _normalize_task_payload_for_snapshot(task_payload)
    if not validate_required_keys(task_payload, ["Task list", "Logic Analysis"]):
        return
    if not _validate_task_payload_paths(task_payload):
        return
    os.makedirs(output_dir, exist_ok=True)
    task_list_path = os.path.join(output_dir, "task_list.json")
    with open(task_list_path, "w", encoding="utf-8") as f:
        json.dump(task_payload, f, ensure_ascii=False, indent=2)


def _write_feature_repo_artifacts(output_dir: str, repo_index: dict, design_payload: dict, task_payload: dict) -> None:
    if not repo_index or not isinstance(task_payload, dict):
        return
    feature_metadata = _build_feature_metadata(design_payload, task_payload)
    closure_payload = build_modification_closure(repo_index, task_payload, feature_metadata=feature_metadata)
    payload_to_save = dict(repo_index)
    payload_to_save["modification_closure"] = {"files": closure_payload.get("modification_closure", {})}
    payload_to_save["context_bundle"] = {"files": closure_payload.get("context_bundle", {})}
    save_repo_index(output_dir, payload_to_save)


def _contains_numbered_python_file_names(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False

    def _walk(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from _walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from _walk(child)
        elif isinstance(value, str):
            yield value

    pattern = re.compile(r"(?:^|[\\/])\d+_[A-Za-z0-9_./\\-]*\.py\b")
    for text in _walk(payload):
        if pattern.search(text.strip()):
            return True
    return False


def run_planning(paper_name: str, gpt_version: str, output_dir: str,
                 paper_format: str = "JSON",
                 pdf_json_path: str = None,
                 pdf_latex_path: str = None,
                 prompt_set: str = None,
                 baseline_repo_dir: str = None,
                 api_predefine_contract_path: str = None) -> None:

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=24000)

    baseline_repo_summary = ""
    repo_manifest_summary = ""
    entrypoint_summary = ""
    repo_index = {}
    if prompt_set == "feature":
        if baseline_repo_dir and os.path.isdir(baseline_repo_dir):
            repo_index = build_repo_index(baseline_repo_dir)
            repo_manifest_summary = summarize_repo_index(repo_index, max_chars=18000)
            entrypoint_summary = summarize_entrypoint_index(repo_index, max_chars=5000)
            baseline_repo_summary = repo_manifest_summary
            save_repo_index(output_dir, repo_index)
        else:
            baseline_repo_summary = format_paper_content_for_prompt(
                _get_repo_content(baseline_repo_dir),
                max_chars=42000,
            )
            repo_manifest_summary = baseline_repo_summary
            entrypoint_summary = "(none)"

    def api_call(msg, gpt_version):
        return chat_completion_with_retry(client, gpt_version, msg)

    system_msg = {'role': "system", "content": render_prompt(
        _prompt_path(prompt_set, "planning_system.txt"),
        paper_format=paper_format)}

    def get_instruction_msg(idx: int, context_lst: list):
        base_kwargs = {
            "paper_content": paper_content_prompt,
            "baseline_repo_summary": baseline_repo_summary,
            "repo_manifest_summary": repo_manifest_summary or baseline_repo_summary,
            "entrypoint_summary": entrypoint_summary,
            "overview": context_lst[0] if len(context_lst) > 0 else "",
            "design": context_lst[1] if len(context_lst) > 1 else "",
            "task": context_lst[2] if len(context_lst) > 2 else "",
        }
        prompt_names = [
            "planning_user_plan.txt",
            "planning_user_design.txt",
            "planning_user_task.txt",
            "planning_user_config.txt",
        ]
        return [{
            "role": "user",
            "content": render_prompt(_prompt_path(prompt_set, prompt_names[idx]), **base_kwargs),
        }]

    responses = []
    trajectories = [system_msg]
    total_accumulated_cost = 0
    for idx in range(4):
        context_lst = [
            item["choices"][0]["message"]["content"]
            for item in responses
            if item.get("choices") and item["choices"][0].get("message", {}).get("content")
        ]
        instruction_msg = get_instruction_msg(idx, context_lst)
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
            if idx in (0, 3):
                break
            model_text = completion_json["choices"][0]["message"]["content"]
            payload = _parse_planning_payload(model_text)
            required = _get_required_keys(prompt_set, idx)
            has_required_keys = validate_required_keys(payload, required)
            has_numbered_names = _contains_numbered_python_file_names(payload)
            has_valid_task_paths = True
            logic_analysis_error = ""
            if idx == 2:
                has_valid_logic_analysis, logic_analysis_error = _validate_logic_analysis_entries(payload)
                has_valid_task_paths = _validate_task_payload_paths(payload)
            else:
                has_valid_logic_analysis = True
            if has_required_keys and not has_numbered_names and has_valid_logic_analysis and has_valid_task_paths:
                break
            trajectories.append({
                "role": "assistant",
                "content": model_text,
            })
            trajectories.append({
                "role": "user",
                "content": (
                    "Your planning output contains forbidden numbered Python file names "
                    "(for example `1_utils.py` or `2_model.py`). Return ONLY corrected [CONTENT]{json}[/CONTENT] "
                    "using descriptive non-numbered snake_case file names."
                    if has_numbered_names else
                    "Your `Logic Analysis` is invalid. Return ONLY corrected [CONTENT]{json}[/CONTENT]. "
                    "Each `Logic Analysis` item must be exactly `[relative_file_path, description]`, "
                    "and the first element must be a single pure relative file path only. "
                    "Do not merge multiple files into one key such as `run.sh and requirements.txt`. "
                    f"Validation detail: {logic_analysis_error}"
                    if not has_valid_logic_analysis else
                    "Your `Task list` is invalid. Return ONLY corrected [CONTENT]{json}[/CONTENT] with `Task list` items "
                    "as pure relative file paths such as `main.py` or `detectors/logprob_detector.py`. "
                    "Do not include directories, absolute paths, `..`, or descriptions like `: implement ...`."
                    if not has_valid_task_paths else
                    "Your output format is invalid. Return ONLY [CONTENT]{json}[/CONTENT] "
                    f"with all required keys: {required}. Do not include any extra text. "
                    + (
                        "For feature planning, explicitly provide replacement anchors, callsite updates, "
                        "public interface changes, and file-naming review fields."
                        if prompt_set == "feature" else
                        "For baseline planning, return strict JSON only and ensure `Task list` and "
                        "`Logic Analysis` are valid structured fields."
                    )
                ),
            })
            logger.warning(f"[Planning] Output schema invalid at stage {idx}, retry {attempt+1}/3")

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})
        if idx == 2:
            task_payload = _parse_planning_payload(message.content)
            _write_task_list_snapshot(output_dir, task_payload)
            design_text = responses[1]["choices"][0]["message"]["content"] if len(responses) > 1 else ""
            design_payload = _parse_planning_payload(design_text)
            if prompt_set == "feature" and repo_index:
                _write_feature_repo_artifacts(output_dir, repo_index, design_payload, task_payload)

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
