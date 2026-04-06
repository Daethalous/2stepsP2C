import json
import os
import re
from typing import Any, NamedTuple, List, Dict, Optional

from core.exceptions import PipelineError
from core.repo_index import REPO_INDEX_FILENAMES
from core.utils import (
    extract_planning,
    content_to_json,
    parse_structured_json,
    try_parse_structured_json,
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
    feature_metadata: Dict[str, Any]
    repo_manifest: Dict[str, Any]
    symbol_index: Dict[str, Any]
    call_graph: Dict[str, Any]
    entrypoint_index: Dict[str, Any]
    modification_closure_by_file: Dict[str, Any]
    context_bundle_by_file: Dict[str, Any]


def _get_key(d: dict, *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _strip_reasoning_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for marker in ("</think>", "</redacted_thinking>"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text.strip()


def _parse_structured_payload(text: str) -> dict:
    payload = parse_structured_json(text)
    if payload:
        return payload
    payload = content_to_json(text)
    if isinstance(payload, dict):
        return payload
    return {}


def _parse_structured_payload_with_status(text: str) -> tuple[dict, bool]:
    payload, ok = try_parse_structured_json(text)
    if payload:
        return payload, ok
    fallback = content_to_json(text)
    if isinstance(fallback, dict) and fallback:
        return fallback, True
    return {}, ok


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
    m = re.match(r"^([A-Za-z0-9_./\\-]+\.[A-Za-z0-9._-]+)\b", cleaned)
    if m:
        cleaned = m.group(1)
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/{2,}", "/", cleaned).strip().lstrip("./")
    return cleaned


def _normalize_task_path_candidate(name: str) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().strip("'\"")
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


def validate_todo_file_path(name: str, source: str = "Task list entry") -> str:
    candidate = _normalize_task_path_candidate(name)
    if not candidate:
        raise PipelineError(f"{source} invalid: empty path.")
    if os.path.isabs(candidate) or re.match(r"^[A-Za-z]:[\\/]", candidate):
        raise PipelineError(f"{source} invalid: '{name}' is not a relative path.")
    if any(token in candidate for token in ("..", "*", "?", "[", "]")):
        raise PipelineError(f"{source} invalid: '{name}' contains forbidden path tokens.")
    if any(token in candidate for token in (":", ",", ";")):
        raise PipelineError(f"{source} invalid: '{name}' contains descriptive separators.")
    if any(ch.isspace() for ch in candidate):
        raise PipelineError(f"{source} invalid: '{name}' contains whitespace and is not a pure file path.")
    if candidate.endswith("/"):
        raise PipelineError(f"{source} invalid: '{name}' must point to a file, not a directory.")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", candidate):
        raise PipelineError(f"{source} invalid: '{name}' contains unsupported path characters.")
    path_parts = candidate.split("/")
    if any(part in ("", ".", "..") for part in path_parts):
        raise PipelineError(f"{source} invalid: '{name}' contains empty or special path segments.")
    if not all(re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in path_parts):
        raise PipelineError(f"{source} invalid: '{name}' contains unsupported path characters.")
    if not _has_generic_file_extension(candidate):
        raise PipelineError(
            f"{source} invalid: '{name}' must be a relative file path with a file extension."
        )
    return candidate


def _stringify_metadata_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (list, tuple)):
        parts = [str(part).strip() for part in item if str(part).strip()]
        return " | ".join(parts)
    if isinstance(item, dict):
        preferred_keys = [
            "target_file",
            "file",
            "owning_file",
            "symbol",
            "change_summary",
            "affected_files",
            "rationale",
            "details",
        ]
        parts = []
        for key in preferred_keys:
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(f"{key}: {text}")
        if parts:
            return "; ".join(parts)
        return "; ".join(
            f"{key}: {value}" for key, value in item.items() if str(value).strip()
        )
    return str(item).strip()


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        normalized = []
        for item in value:
            text = _stringify_metadata_item(item)
            if text:
                normalized.append(text)
        return normalized
    text = _stringify_metadata_item(value)
    return [text] if text else []


def _normalize_string_matrix(value: Any) -> List[List[str]]:
    normalized: List[List[str]] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if isinstance(item, (list, tuple)):
            row = [str(part).strip() for part in item if str(part).strip()]
            if row:
                normalized.append(row)
        elif isinstance(item, dict):
            row = []
            for key in (
                "target_file",
                "file",
                "owning_file",
                "target_symbol",
                "symbol",
                "replacement_mode",
                "change_summary",
                "affected_files",
                "rationale",
                "details",
            ):
                value_part = item.get(key)
                if value_part is None:
                    continue
                text = str(value_part).strip()
                if text:
                    row.append(text)
            if row:
                normalized.append(row)
        else:
            text = str(item).strip()
            if text:
                normalized.append([text])
    return normalized


def _extract_file_keys_from_text(text: str) -> List[str]:
    matches = re.findall(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9._-]+", text or "")
    normalized = []
    seen = set()
    for match in matches:
        file_key = sanitize_todo_file_name(match)
        if not file_key or file_key in seen:
            continue
        normalized.append(file_key)
        seen.add(file_key)
    return normalized


def _append_unique(mapping: Dict[str, List[str]], file_key: str, text: str) -> None:
    if not file_key or not text:
        return
    bucket = mapping.setdefault(file_key, [])
    if text not in bucket:
        bucket.append(text)


def _build_feature_metadata(design_payload: dict, task_payload: dict) -> Dict[str, Any]:
    design_payload = design_payload if isinstance(design_payload, dict) else {}
    task_payload = task_payload if isinstance(task_payload, dict) else {}

    core_replacement_targets = _normalize_string_matrix(
        _get_key(
            design_payload,
            "Core Replacement Targets",
            "core_replacement_targets",
            "Core replacement targets",
        )
    )
    new_files_justification = _normalize_string_matrix(
        _get_key(
            design_payload,
            "New Files Justification",
            "new_files_justification",
            "New files justification",
        )
    )
    file_naming_review = str(
        _get_key(
            design_payload,
            "File Naming Review",
            "file_naming_review",
            "File naming review",
        )
        or ""
    ).strip()
    primary_entry_points = _normalize_string_list(
        _get_key(
            design_payload,
            "Primary Entry Points",
            "primary_entry_points",
            "Primary entry points",
        )
    )
    execution_chain = _normalize_string_list(
        _get_key(
            design_payload,
            "Execution Chain",
            "execution_chain",
            "Execution chain",
        )
    )
    registry_factory_touchpoints = _normalize_string_list(
        _get_key(
            design_payload,
            "Registry/Factory Touchpoints",
            "registry_factory_touchpoints",
            "Registry factory touchpoints",
        )
    )

    callsite_update_list = _normalize_string_matrix(
        _get_key(
            task_payload,
            "Callsite Update List",
            "callsite_update_list",
            "Callsite update list",
        )
    )
    public_interface_changes = _normalize_string_matrix(
        _get_key(
            task_payload,
            "Public Interface Changes",
            "public_interface_changes",
            "Public interface changes",
        )
    )
    forbidden_file_names = _normalize_string_list(
        _get_key(
            task_payload,
            "Forbidden File Names",
            "forbidden_file_names",
            "Forbidden file names",
        )
    )
    modification_closure = _get_key(
        task_payload,
        "Modification Closure",
        "modification_closure",
        "Modification closure",
    )
    if not isinstance(modification_closure, list):
        modification_closure = []

    callsite_updates_by_file: Dict[str, List[str]] = {}
    for item in callsite_update_list:
        if not item:
            continue
        owner = sanitize_todo_file_name(item[0])
        details = " | ".join(item[1:]).strip() if len(item) > 1 else item[0]
        if owner:
            _append_unique(callsite_updates_by_file, owner, details)

    public_interface_changes_by_file: Dict[str, List[str]] = {}
    for item in public_interface_changes:
        detail_text = " | ".join(item).strip()
        if not detail_text:
            continue
        affected_files = []
        if len(item) >= 3:
            affected_files.extend(_extract_file_keys_from_text(item[2]))
        if not affected_files:
            affected_files.extend(_extract_file_keys_from_text(detail_text))
        for file_key in affected_files:
            _append_unique(public_interface_changes_by_file, file_key, detail_text)

    new_files_by_path: Dict[str, List[str]] = {}
    for item in new_files_justification:
        if not item:
            continue
        new_file = sanitize_todo_file_name(item[0])
        detail_text = " | ".join(item[1:]).strip() if len(item) > 1 else ""
        if new_file:
            _append_unique(new_files_by_path, new_file, detail_text)

    return {
        "core_replacement_targets": core_replacement_targets,
        "new_files_justification": new_files_justification,
        "file_naming_review": file_naming_review,
        "primary_entry_points": primary_entry_points,
        "execution_chain": execution_chain,
        "registry_factory_touchpoints": registry_factory_touchpoints,
        "callsite_update_list": callsite_update_list,
        "public_interface_changes": public_interface_changes,
        "forbidden_file_names": forbidden_file_names,
        "modification_closure": modification_closure,
        "callsite_updates_by_file": callsite_updates_by_file,
        "public_interface_changes_by_file": public_interface_changes_by_file,
        "new_files_by_path": new_files_by_path,
    }


def _load_planning_context_texts(output_dir: str) -> list:
    """Prefer exact stage outputs from planning_response.json; fallback to legacy trajectories."""
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
                    return context_lst[:3]
        except Exception:
            pass
    return extract_planning(f'{output_dir}/planning_trajectories.json')


def _load_optional_json(output_dir: str, file_name: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, file_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_repo_index_artifacts(output_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        key: _load_optional_json(output_dir, file_name)
        for key, file_name in REPO_INDEX_FILENAMES.items()
        if key in {"repo_manifest", "symbol_index", "call_graph", "entrypoint_index"}
    }


def load_modification_closure(output_dir: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    modification_closure = _load_optional_json(output_dir, REPO_INDEX_FILENAMES["modification_closure"])
    context_bundle = _load_optional_json(output_dir, REPO_INDEX_FILENAMES["context_bundle"])
    return (
        modification_closure.get("files", {}) if isinstance(modification_closure, dict) else {},
        context_bundle.get("files", {}) if isinstance(context_bundle, dict) else {},
    )


def _convert_feature_logic_task_list_to_legacy(task_payload: dict):
    """Convert older feature-only task schema into legacy/ref todo outputs."""
    if not isinstance(task_payload, dict):
        return None, None

    logic_task_list = _get_key(task_payload, "Logic_Task_List", "logic_task_list")
    if not isinstance(logic_task_list, list):
        return None, None

    calibration_notes = _get_key(
        task_payload, "Secondary_Calibration_Notes", "secondary_calibration_notes"
    )
    if not isinstance(calibration_notes, list):
        calibration_notes = []

    calibration_summary_lines = []
    for note in calibration_notes:
        if not isinstance(note, dict):
            continue
        component = str(note.get("component", "")).strip()
        upgrade = str(note.get("upgraded_paper_logic", "")).strip()
        reason = str(note.get("reason_for_upgrade", "")).strip()
        parts = [x for x in (component, upgrade, reason) if x]
        if parts:
            calibration_summary_lines.append("- " + " | ".join(parts))

    todo_file_lst = []
    logic_analysis = []
    logic_analysis_dict = {}

    for item in logic_task_list:
        if not isinstance(item, dict):
            continue
        raw_target_file = str(item.get("target_file", ""))
        target_file = validate_todo_file_path(raw_target_file, "Task list entry")
        if not target_file:
            continue

        atomic_steps = item.get("atomic_steps", [])
        dependency_imports = item.get("dependency_imports", [])

        summary_lines = []
        if calibration_summary_lines:
            summary_lines.append("Secondary calibration notes:")
            summary_lines.extend(calibration_summary_lines)

        if isinstance(atomic_steps, list) and atomic_steps:
            summary_lines.append("Atomic steps:")
            for idx, step in enumerate(atomic_steps, start=1):
                if not isinstance(step, dict):
                    continue
                step_id = str(step.get("step_id", f"step_{idx}")).strip()
                description = str(step.get("description", "")).strip()
                math_logic = str(step.get("math_logic", "")).strip()
                expected_input = str(step.get("expected_input_shape", "")).strip()
                expected_output = str(step.get("expected_output_shape", "")).strip()
                summary_lines.append(f"{idx}. [{step_id}] {description}")
                if math_logic:
                    summary_lines.append(f"   Math: {math_logic}")
                if expected_input or expected_output:
                    summary_lines.append(
                        f"   Shapes: {expected_input or '(unknown)'} -> {expected_output or '(unknown)'}"
                    )

        if isinstance(dependency_imports, list) and dependency_imports:
            summary_lines.append("Dependency imports:")
            summary_lines.extend(f"- {dep}" for dep in dependency_imports if str(dep).strip())

        summary_text = "\n".join(summary_lines).strip()
        if not summary_text:
            summary_text = "(no detailed logic analysis available)"

        todo_file_lst.append(target_file)
        logic_analysis.append([target_file, summary_text])
        logic_analysis_dict[target_file] = summary_text

    if not logic_analysis_dict:
        return None, None

    legacy_payload = dict(task_payload)
    legacy_payload["Task list"] = todo_file_lst
    legacy_payload["Logic Analysis"] = logic_analysis
    return legacy_payload, logic_analysis_dict


def _summarize_task_entry(entry: dict, global_logic_analysis: str = "") -> str:
    """Build a per-file analysis summary from task-entry dicts."""
    if not isinstance(entry, dict):
        return ""

    summary_lines = []
    global_logic_analysis = str(global_logic_analysis or "").strip()
    if global_logic_analysis:
        summary_lines.append("Global logic analysis:")
        summary_lines.append(global_logic_analysis)

    step_list = entry.get("steps")
    if not isinstance(step_list, list):
        step_list = entry.get("atomic_steps")
    if isinstance(step_list, list) and step_list:
        summary_lines.append("File-specific steps:")
        for idx, step in enumerate(step_list, start=1):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id", step.get("step_id", f"step_{idx}"))).strip()
            instruction = str(step.get("instruction", step.get("description", ""))).strip()
            math_logic = str(step.get("math", step.get("math_logic", ""))).strip()
            in_shape = str(step.get("in_shape", step.get("expected_input_shape", ""))).strip()
            out_shape = str(step.get("out_shape", step.get("expected_output_shape", ""))).strip()
            summary_lines.append(f"{idx}. [{step_id}] {instruction}")
            if math_logic:
                summary_lines.append(f"   Math: {math_logic}")
            if in_shape or out_shape:
                summary_lines.append(
                    f"   Shapes: {in_shape or '(unknown)'} -> {out_shape or '(unknown)'}"
                )

    imports = entry.get("imports")
    if not isinstance(imports, list):
        imports = entry.get("dependency_imports")
    if isinstance(imports, list) and imports:
        summary_lines.append("Dependency imports:")
        summary_lines.extend(f"- {item}" for item in imports if str(item).strip())

    replaced_symbols = _normalize_string_list(
        _get_key(entry, "replaced_symbols", "modified_symbols", "symbols")
    )
    if replaced_symbols:
        summary_lines.append("Modified symbols:")
        summary_lines.extend(f"- {item}" for item in replaced_symbols)

    callsite_updates = _normalize_string_list(
        _get_key(entry, "callsite_updates", "callsite_update_list")
    )
    if callsite_updates:
        summary_lines.append("Callsite updates:")
        summary_lines.extend(f"- {item}" for item in callsite_updates)

    public_changes = _normalize_string_list(
        _get_key(entry, "public_interface_changes", "interface_changes")
    )
    if public_changes:
        summary_lines.append("Public interface changes:")
        summary_lines.extend(f"- {item}" for item in public_changes)

    return "\n".join(summary_lines).strip()


def load_pipeline_context(output_dir: str) -> PipelineContext:
    with open(f'{output_dir}/planning_config.yaml', encoding="utf-8") as f:
        config_yaml = f.read()

    context_lst = _load_planning_context_texts(output_dir)
    design_payload = _parse_structured_payload(context_lst[1]) if len(context_lst) > 1 else {}

    if os.path.exists(f'{output_dir}/task_list.json'):
        with open(f'{output_dir}/task_list.json', encoding="utf-8") as f:
            task_list = json.load(f)
        task_parse_ok = True
    else:
        task_source_text = context_lst[2] if len(context_lst) > 2 else ""
        task_list, task_parse_ok = _parse_structured_payload_with_status(task_source_text)

    feature_metadata = _build_feature_metadata(design_payload, task_list)
    repo_index_artifacts = load_repo_index_artifacts(output_dir)
    modification_closure_by_file, context_bundle_by_file = load_modification_closure(output_dir)

    has_legacy_schema = validate_required_keys(task_list, ["Task list", "Logic Analysis"])

    if not has_legacy_schema:
        converted_task_list, converted_logic_analysis_dict = _convert_feature_logic_task_list_to_legacy(task_list)
        if converted_task_list is not None:
            task_list = converted_task_list
        elif not isinstance(task_list, dict):
            task_list = {}
        else:
            converted_logic_analysis_dict = None
    else:
        converted_logic_analysis_dict = None

    raw_todo_file_lst = _get_key(task_list, 'Task list', 'task_list', 'task list', 'Task List')

    logic_analysis = _get_key(task_list, 'Logic Analysis', 'logic_analysis', 'logic analysis', 'Logic analysis')
    if logic_analysis is None:
        if not task_parse_ok or not isinstance(task_list, dict) or len(task_list) == 0:
            raise PipelineError(
                "Planning task payload is malformed and could not be parsed into structured JSON. "
                f"Inspect '{output_dir}/planning_response.json' stage-2 task output or regenerate planning artifacts."
            )
        raise PipelineError(
            "Planning task schema is unsupported. Neither legacy keys "
            "('Task list' / 'Logic Analysis') nor the older feature-only key "
            "'Logic_Task_List' were found."
        )

    if raw_todo_file_lst is None:
        # Fallback for imperfect planning outputs: derive todo files from logic-analysis entries.
        todo_file_lst = []
        for desc in logic_analysis:
            if isinstance(desc, (list, tuple)) and len(desc) > 0:
                candidate = validate_todo_file_path(str(desc[0]), "Task list entry")
                if candidate:
                    todo_file_lst.append(candidate)
        if len(todo_file_lst) == 0:
            raise PipelineError("'Task list' does not exist and no fallback could be extracted from 'Logic Analysis'.")
    else:
        todo_file_lst = []
        for item in raw_todo_file_lst:
            if isinstance(item, dict):
                raw_candidate = str(item.get("file", item.get("target_file", "")))
            else:
                raw_candidate = str(item)
            candidate = validate_todo_file_path(raw_candidate, "Task list entry")
            if candidate:
                todo_file_lst.append(candidate)

    normalized_todo = []
    seen_todo = set()
    for item in todo_file_lst:
        clean_item = validate_todo_file_path(str(item), "Task list entry")
        if not clean_item or clean_item in seen_todo:
            continue
        normalized_todo.append(clean_item)
        seen_todo.add(clean_item)
    todo_file_lst = normalized_todo

    logic_analysis_dict = {}
    if converted_logic_analysis_dict is not None:
        logic_analysis_dict.update(converted_logic_analysis_dict)
    elif isinstance(logic_analysis, str) and isinstance(raw_todo_file_lst, list):
        for item in raw_todo_file_lst:
            if not isinstance(item, dict):
                continue
            file_key = validate_todo_file_path(
                str(item.get("file", item.get("target_file", ""))),
                "Task list entry",
            )
            if not file_key:
                continue
            summary = _summarize_task_entry(item, global_logic_analysis=logic_analysis)
            logic_analysis_dict[file_key] = summary or logic_analysis
    else:
        for desc in logic_analysis:
            if not isinstance(desc, (list, tuple)) or len(desc) < 2:
                continue
            raw_file_key = str(desc[0])
            try:
                file_key = validate_todo_file_path(raw_file_key, "Logic Analysis file entry")
            except PipelineError as exc:
                raise PipelineError(
                    f"{exc} Logic Analysis file entry must be a single relative file path. "
                    "Do not merge multiple files into one key such as "
                    "`run.sh and requirements.txt`."
                ) from exc
            if not file_key:
                continue
            logic_analysis_dict[file_key] = desc[1]

    return PipelineContext(
        config_yaml=config_yaml,
        context_lst=context_lst,
        task_list=task_list,
        todo_file_lst=todo_file_lst,
        logic_analysis_dict=logic_analysis_dict,
        feature_metadata=feature_metadata,
        repo_manifest=repo_index_artifacts.get("repo_manifest", {}),
        symbol_index=repo_index_artifacts.get("symbol_index", {}),
        call_graph=repo_index_artifacts.get("call_graph", {}),
        entrypoint_index=repo_index_artifacts.get("entrypoint_index", {}),
        modification_closure_by_file=modification_closure_by_file,
        context_bundle_by_file=context_bundle_by_file,
    )
