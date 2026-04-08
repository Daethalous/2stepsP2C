import json
import os
import time
from tqdm import tqdm
import re
import copy
import hashlib
from typing import Any
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.exceptions import PipelineError
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.repo_index import collect_context_bundle, render_context_bundle_for_prompt
from core.utils import (extract_code_from_content, print_response, print_log_cost,
                        load_accumulated_cost, save_accumulated_cost,
                        read_python_files,
                        format_paper_content_for_prompt,
                        build_code_interface_summary,
                        contains_forbidden_placeholders)
import argparse

logger = get_logger(__name__)
DEBUG_LOG_PATH = "debug-084f81.log"
FEATURE_CODE_FILES_MAX_CHARS = 12000
FEATURE_BASELINE_FILE_MAX_CHARS = 12000
FEATURE_ANALYSIS_MAX_CHARS = 10000
FEATURE_API_CONTRACT_MAX_CHARS = 8000
BASELINE_INTERFACE_SUMMARY_MAX_CHARS = 14000
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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


def _sanitize_prompt_text(text: Any, max_chars: int = None) -> str:
    if text is None:
        sanitized = ""
    elif isinstance(text, str):
        sanitized = text
    else:
        sanitized = str(text)
    sanitized = sanitized.encode("utf-8", errors="replace").decode("utf-8")
    sanitized = sanitized.replace("\x00", "")
    sanitized = CONTROL_CHAR_RE.sub(" ", sanitized)
    if max_chars is not None and len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars] + "\n...(truncated for token budget)..."
    return sanitized


def _sanitize_payload(obj: Any):
    if isinstance(obj, str):
        return _sanitize_prompt_text(obj)
    if isinstance(obj, list):
        return [_sanitize_payload(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _sanitize_payload(value) for key, value in obj.items()}
    return obj


def _prepare_messages_for_api(messages: list, todo_file_name: str, segment_meta: dict) -> tuple[list, int]:
    safe_messages = _sanitize_payload(messages)
    dumped = json.dumps(safe_messages, ensure_ascii=False)
    dumped.encode("utf-8", errors="strict")
    _debug_log(
        run_id="initial",
        hypothesis_id="H2",
        location="workflow/coding.py:before_api_call",
        message="prepared trajectories",
        data={
            "todo_file_name": todo_file_name,
            "msg_count": len(safe_messages),
            "roles": [m.get("role") for m in safe_messages],
            "content_types": [type(m.get("content")).__name__ for m in safe_messages],
            "payload_len": len(dumped),
            "segment_meta": segment_meta,
        },
    )
    return safe_messages, len(dumped)


def _make_safe_artifact_stem(path_text: str) -> str:
    normalized = sanitize_todo_file_name(path_text) or str(path_text or "").strip()
    normalized = normalized.replace("\\", "/")
    readable = normalized.replace("/", "__")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("._")
    readable = re.sub(r"_+", "_", readable)
    if not readable:
        readable = "artifact"
    suffix = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def _ensure_path_within_root(root_dir: str, target_path: str) -> str:
    root_abs = os.path.abspath(root_dir)
    target_abs = os.path.abspath(target_path)
    try:
        within_root = os.path.commonpath([root_abs, target_abs]) == root_abs
    except ValueError:
        within_root = False
    if not within_root:
        raise PipelineError(
            f"Refusing to write outside repo boundary: '{target_path}' is not inside '{root_dir}'."
        )
    return target_abs


def _load_global_api_contract_stub(output_dir: str, explicit_path: str = None) -> str:
    contract_path = explicit_path or os.path.join(output_dir, "api_predefine_contract.pyi")
    if not os.path.exists(contract_path):
        return "(none)"
    try:
        with open(contract_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return _sanitize_prompt_text(text if text else "(none)", max_chars=FEATURE_API_CONTRACT_MAX_CHARS)
    except Exception:
        return "(none)"


def _build_done_code_context(done_file_dict: dict, done_file_lst: list, max_total_chars: int = 20000) -> str:
    parts = []
    total = 0
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"):
            continue
        code = done_file_dict.get(done_file, "")
        if not code:
            continue
        chunk = f"\n### {done_file}\n```python\n{code}\n```\n"
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 400:
                parts.append(chunk[:remaining] + "\n...(truncated for token budget)...")
            else:
                parts.append("\n...(truncated for token budget)...")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts) if parts else "(no previous code files)"


def run_coding(paper_name: str, gpt_version: str, output_dir: str,
               output_repo_dir: str,
               paper_format: str = "JSON",
               pdf_json_path: str = None,
               pdf_latex_path: str = None,
               prompt_set: str = None,
               baseline_repo_dir: str = None,
               live_repo_dir: str = None,
               api_predefine_contract_path: str = None) -> None:
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
    paper_content_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(paper_content, max_chars=22000),
        max_chars=22000,
    )
    ctx = load_pipeline_context(output_dir)
    config_yaml = _sanitize_prompt_text(ctx.config_yaml, max_chars=16000)
    context_lst = ctx.context_lst
    feature_metadata = getattr(ctx, "feature_metadata", {}) or {}
    repo_index = {
        "repo_manifest": ctx.repo_manifest,
        "symbol_index": ctx.symbol_index,
        "call_graph": ctx.call_graph,
        "entrypoint_index": ctx.entrypoint_index,
    }
    overview_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[0], max_chars=12000), max_chars=12000
    ) if len(context_lst) > 0 else ""
    design_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[1], max_chars=16000), max_chars=16000
    ) if len(context_lst) > 1 else ""
    task_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[2], max_chars=16000), max_chars=16000
    ) if len(context_lst) > 2 else ""
    global_api_contract_stub = _load_global_api_contract_stub(
        output_dir, explicit_path=api_predefine_contract_path
    )
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

    def _get_feature_closure(path_name: str) -> dict:
        clean_name = sanitize_todo_file_name(path_name)
        closure = ctx.modification_closure_by_file.get(clean_name, {}) if clean_name else {}
        if closure:
            return closure
        return {"path": clean_name or path_name}

    def _is_entry_point_file(path_name: str) -> bool:
        clean_name = sanitize_todo_file_name(path_name)
        if clean_name in set(ctx.entrypoint_index.get("primary_files", [])):
            return True
        closure = _get_feature_closure(clean_name)
        if closure.get("entrypoints"):
            return True
        lowered = clean_name.lower()
        return any(tag in lowered for tag in ("main.py", "__init__.py", "factory", "registry", "runner", "train"))

    def _is_numbered_python_file(path_name: str) -> bool:
        base_name = os.path.basename(path_name.replace("\\", "/"))
        return bool(re.match(r"^\d+_.*\.py$", base_name))

    def _contains_numbered_module_imports(code_text: str) -> bool:
        return bool(
            re.search(
                r"^\s*(?:from|import)\s+[0-9][A-Za-z0-9_\.]*\b",
                code_text,
                flags=re.MULTILINE,
            )
        )

    def _violates_forbidden_python_file_name(path_name: str) -> bool:
        normalized = path_name.replace("\\", "/")
        base_name = os.path.basename(normalized)
        if _is_numbered_python_file(base_name):
            return True
        forbidden_names = feature_metadata.get("forbidden_file_names", [])
        for raw_name in forbidden_names:
            name = str(raw_name).strip()
            if not name:
                continue
            if name == "<number>_*.py" and _is_numbered_python_file(base_name):
                return True
            clean_name = sanitize_todo_file_name(name)
            if clean_name and normalized == clean_name:
                return True
            if name == base_name:
                return True
        return False

    def _extract_new_definitions(code_text: str):
        matches = re.findall(r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", code_text, flags=re.MULTILINE)
        return matches[:20]

    def _find_analysis_response_path(todo_file_name: str):
        for candidate in _resolve_todo_candidates(todo_file_name):
            safe_name = _make_safe_artifact_stem(candidate)
            path = os.path.join(output_dir, f"{safe_name}_simple_analysis_response.json")
            legacy_path = os.path.join(output_dir, f"{candidate.replace('/', '_')}_simple_analysis_response.json")
            if os.path.exists(path):
                return path
            if os.path.exists(legacy_path):
                return legacy_path
        return None

    def _load_feature_file_code(todo_file_name: str):
        for candidate in _resolve_todo_candidates(todo_file_name):
            if output_repo_dir:
                live_path = os.path.join(output_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live"
            if live_repo_dir:
                live_path = os.path.join(live_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live_repo"
            if baseline_repo_dir:
                baseline_path = os.path.join(baseline_repo_dir, candidate)
                if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as bf:
                        return bf.read(), "baseline"
        return None, "new"

    def _build_context_bundle_for_file(todo_file_name: str) -> dict:
        if not ctx.repo_manifest:
            return {}
        closure = _get_feature_closure(todo_file_name)
        return collect_context_bundle(
            repo_index,
            closure,
            primary_repo_dir=baseline_repo_dir,
            secondary_repo_dir=output_repo_dir,
        )

    def _build_feature_integration_hint(todo_file_name: str) -> str:
        hint_sections = []
        closure = _get_feature_closure(todo_file_name)

        replacement_targets = feature_metadata.get("core_replacement_targets", [])
        matching_targets = []
        for row in replacement_targets:
            if not row:
                continue
            target_file = sanitize_todo_file_name(str(row[0]))
            if target_file == todo_file_name:
                matching_targets.append(" | ".join(str(part).strip() for part in row[1:] if str(part).strip()))
        if matching_targets:
            hint_sections.append(
                "Planned replacement anchors:\n" + "\n".join(f"- {item}" for item in matching_targets)
            )

        callsite_updates = feature_metadata.get("callsite_updates_by_file", {}).get(todo_file_name, [])
        if callsite_updates:
            hint_sections.append(
                "Required callsite updates:\n" + "\n".join(f"- {item}" for item in callsite_updates)
            )

        public_changes = feature_metadata.get("public_interface_changes_by_file", {}).get(todo_file_name, [])
        if public_changes:
            hint_sections.append(
                "Public interface changes touching this file:\n" + "\n".join(f"- {item}" for item in public_changes)
            )

        new_file_reasons = feature_metadata.get("new_files_by_path", {}).get(todo_file_name, [])
        if new_file_reasons:
            hint_sections.append(
                "Justified new-file rationale:\n" + "\n".join(f"- {item}" for item in new_file_reasons)
            )

        file_naming_review = str(feature_metadata.get("file_naming_review", "")).strip()
        forbidden_names = feature_metadata.get("forbidden_file_names", [])
        if file_naming_review or forbidden_names:
            naming_lines = []
            if file_naming_review:
                naming_lines.append(file_naming_review)
            naming_lines.extend(f"- {item}" for item in forbidden_names[:20] if str(item).strip())
            hint_sections.append("Naming constraints:\n" + "\n".join(naming_lines))

        if closure.get("entrypoints"):
            hint_sections.append(
                "Resolved entrypoint chain:\n" + "\n".join(f"- {item}" for item in closure.get("entrypoints", []))
            )
        if closure.get("synchronized_edits"):
            hint_sections.append(
                "Resolved synchronized edit targets:\n"
                + "\n".join(f"- {item}" for item in closure.get("synchronized_edits", []))
            )
        if closure.get("target_symbols"):
            hint_sections.append(
                "Resolved target symbols:\n" + "\n".join(f"- {item}" for item in closure.get("target_symbols", []))
            )

        if _is_entry_point_file(todo_file_name):
            entry_lines = [
                "Verify the main execution chain, trainer/evaluator wiring, and any registry or factory callbacks.",
                "Ensure the final code path really reaches the paper logic rather than a side module.",
            ]
            if new_definitions:
                entry_lines.extend(f"- {item}" for item in new_definitions[:50])
            hint_sections.append("Entrypoint reintegration checks:\n" + "\n".join(entry_lines))

        return "\n\n".join(section for section in hint_sections if section).strip()

    def _sort_feature_todo_files(file_list: list) -> list:
        def _priority(path_name: str) -> tuple:
            closure = _get_feature_closure(path_name)
            role_tags = closure.get("focus_role_tags", [])
            if any(tag in role_tags for tag in ("config",)):
                return (0, path_name)
            if closure.get("target_symbols"):
                return (1, path_name)
            if any(tag in role_tags for tag in ("registry", "factory", "entrypoint", "main", "train", "eval")):
                return (3, path_name)
            if path_name.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh")):
                return (4, path_name)
            return (2, path_name)
        return sorted(file_list, key=_priority)

    def _normalize_expected_symbol(raw_symbol: str) -> str:
        text = str(raw_symbol or "").strip()
        if not text:
            return ""

        if "." in text:
            text = text.split(".")[-1].strip()

        if "(" in text:
            prefix = text.split("(", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", prefix):
                return prefix

        direct_match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if direct_match:
            return text

        leading_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", text)
        if not leading_match:
            return ""

        candidate = leading_match.group(1)
        trailing = text[leading_match.end():].strip()
        if not trailing:
            return candidate

        # Accept simple annotation tails like "(base)" but ignore broader prose such as
        # "Detector (base class) and factory", which is not a stable code symbol.
        if trailing.startswith("(") and " and " not in trailing.lower() and "," not in trailing and "|" not in trailing:
            return candidate
        return ""

    def _extract_python_defs(code_text: str) -> dict:
        classes = set(re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", code_text, flags=re.MULTILINE))
        functions = set(re.findall(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", code_text, flags=re.MULTILINE))
        return {"classes": classes, "functions": functions}

    def _get_feature_validation_error(todo_file_name: str, code_text: str) -> str:
        if not todo_file_name.endswith(".py"):
            return ""
        closure = _get_feature_closure(todo_file_name)
        defs = _extract_python_defs(code_text)
        missing_symbols = []
        for symbol in closure.get("target_symbols", []):
            normalized = _normalize_expected_symbol(symbol)
            if not normalized:
                continue
            if normalized[:1].isupper():
                matched = normalized in defs["classes"] or bool(
                    re.search(rf"\b{re.escape(normalized)}\b", code_text)
                )
            else:
                matched = normalized in defs["functions"] or bool(
                    re.search(rf"\b{re.escape(normalized)}\b", code_text)
                )
            if not matched:
                missing_symbols.append(symbol)
        if missing_symbols:
            return (
                f"Generated code for {todo_file_name} does not contain expected target symbols: "
                f"{missing_symbols[:8]}"
            )
        return ""

    def _validate_feature_generation(todo_file_name: str, code_text: str) -> None:
        feature_error = _get_feature_validation_error(todo_file_name, code_text)
        if feature_error:
            raise PipelineError(feature_error)

    def _is_python_source_file(todo_file_name: str) -> bool:
        return str(todo_file_name or "").strip().lower().endswith(".py")

    def _validate_generated_code_attempt(todo_file_name: str, code_text: str) -> tuple[bool, str, str]:
        if contains_forbidden_placeholders(code_text):
            return (
                False,
                "unfinished_placeholders",
                "Generated code still contains unfinished implementation markers "
                "(for example NotImplementedError/TODO/stub).",
            )
        if not _is_python_source_file(todo_file_name):
            return True, "", ""
        if _contains_numbered_module_imports(code_text):
            return (
                False,
                "numbered_imports",
                "Generated code still contains numbered Python module imports.",
            )
        try:
            compile(code_text, todo_file_name, "exec")
        except SyntaxError as exc:
            return False, "syntax_error", f"Generated code for {todo_file_name} is syntactically invalid: {exc}"
        if prompt_set == "feature":
            feature_error = _get_feature_validation_error(todo_file_name, code_text)
            if feature_error:
                return False, "feature_validation_error", feature_error
        return True, "", ""

    def _build_retry_message(error_kind: str, error_message: str) -> str:
        if error_kind == "numbered_imports":
            return (
                "Output rejected: code must not create or import numbered Python modules "
                "(for example `from 1_utils import ...`). "
                "Use descriptive non-numbered module names instead and return ONLY the corrected complete file code."
            )
        if error_kind == "syntax_error":
            return (
                "Output rejected: the generated code has a Python syntax error.\n"
                f"Compiler error: {error_message}\n"
                "Return ONLY the fully corrected complete file code. Do not explain. Do not provide a diff."
            )
        if error_kind == "feature_validation_error":
            return (
                "Output rejected: the generated code failed feature integration validation.\n"
                f"Validation detail: {error_message}\n"
                "Return ONLY the fully corrected complete file code, keeping the known target symbols/callsites consistent."
            )
        return (
            "Output rejected: code still contains unfinished implementation markers "
            "(for example NotImplementedError/TODO/stub). "
            "Return ONLY the complete corrected file code, no explanations."
        )

    if prompt_set == "feature":
        existing = read_python_files(output_repo_dir)
        for fname, content in existing.items():
            if fname not in done_file_dict:
                done_file_dict[fname] = content
        todo_file_lst = _sort_feature_todo_files(todo_file_lst)

    code_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "coding_system.txt"),
            paper_format=paper_format)}]
    code_msg = _sanitize_payload(code_msg)

    def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst): 
        if prompt_set == "feature":
            context_bundle = _build_context_bundle_for_file(todo_file_name)
            bundle_prompt = render_context_bundle_for_prompt(context_bundle) if context_bundle else {}
            if bundle_prompt:
                code_files = _sanitize_prompt_text(
                    bundle_prompt.get("shared_interfaces_code", "(none)"),
                    max_chars=FEATURE_CODE_FILES_MAX_CHARS,
                )
            else:
                code_files = _sanitize_prompt_text(
                    _build_done_code_context(
                        done_file_dict,
                        done_file_lst,
                        max_total_chars=FEATURE_CODE_FILES_MAX_CHARS,
                    ),
                    max_chars=FEATURE_CODE_FILES_MAX_CHARS,
                )
        else:
            code_files = _sanitize_prompt_text(
                build_code_interface_summary(
                    done_file_dict,
                    done_file_lst,
                    max_total_chars=BASELINE_INTERFACE_SUMMARY_MAX_CHARS,
                ),
                max_chars=BASELINE_INTERFACE_SUMMARY_MAX_CHARS,
            )

        extra_kwargs = {}
        if prompt_set == "feature":
            extra_kwargs["integration_hint"] = _build_feature_integration_hint(todo_file_name)
            context_bundle = _build_context_bundle_for_file(todo_file_name)
            bundle_prompt = render_context_bundle_for_prompt(context_bundle) if context_bundle else {}
            if bundle_prompt:
                focus_file_code = bundle_prompt.get("focus_file_code", "(new file — no baseline/live code)")
                extra_kwargs["focus_file_code"] = focus_file_code
                extra_kwargs["required_context_code"] = bundle_prompt.get("required_context_code", "(none)")
                extra_kwargs["upstream_callers_code"] = bundle_prompt.get("upstream_callers_code", "(none)")
                extra_kwargs["downstream_callees_code"] = bundle_prompt.get("downstream_callees_code", "(none)")
                extra_kwargs["shared_interfaces_code"] = bundle_prompt.get("shared_interfaces_code", "(none)")
                extra_kwargs["config_and_registry_code"] = bundle_prompt.get("config_and_registry_code", "(none)")
                extra_kwargs["optional_related_code"] = bundle_prompt.get("optional_related_code", "(none)")
                extra_kwargs["entrypoint_chain"] = bundle_prompt.get("entrypoint_chain", "(none)")
                extra_kwargs["synchronized_edit_targets"] = bundle_prompt.get("synchronized_edit_targets", "(none)")
                extra_kwargs["interface_constraints"] = bundle_prompt.get("interface_constraints", "(none)")
                extra_kwargs["target_symbols"] = bundle_prompt.get("target_symbols", "(none)")
                extra_kwargs["repo_primary_entrypoints"] = bundle_prompt.get("repo_primary_entrypoints", "(none)")
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {bundle_prompt.get('focus_file_source', 'new')}\n"
                    f"{focus_file_code}"
                )
            else:
                extra_kwargs["baseline_file_code"] = "(new file — no baseline/live code)"
                file_code, code_source = _load_feature_file_code(todo_file_name)
                if file_code is None:
                    existing_files = sorted(list(done_file_dict.keys()))
                    extra_kwargs["baseline_file_code"] = (
                        "(new file — no baseline/live code)\n"
                        + ("\nExisting repo files:\n" + "\n".join(f"- {x}" for x in existing_files[:200]) if existing_files else "")
                    )
                    extra_kwargs["focus_file_code"] = "(new file — no baseline/live code)"
                else:
                    extra_kwargs["focus_file_code"] = file_code
                    extra_kwargs["baseline_file_code"] = f"### Source: {code_source}\n{file_code}"
                for key in (
                    "required_context_code",
                    "upstream_callers_code",
                    "downstream_callees_code",
                    "shared_interfaces_code",
                    "config_and_registry_code",
                    "optional_related_code",
                    "entrypoint_chain",
                    "synchronized_edit_targets",
                    "interface_constraints",
                    "target_symbols",
                    "repo_primary_entrypoints",
                ):
                    extra_kwargs.setdefault(key, "(none)")
            extra_kwargs["integration_hint"] = _sanitize_prompt_text(
                extra_kwargs["integration_hint"], max_chars=2600
            )

        sanitized_analysis = _sanitize_prompt_text(
            detailed_logic_analysis,
            max_chars=FEATURE_ANALYSIS_MAX_CHARS if prompt_set == "feature" else 12000,
        )

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
            detailed_logic_analysis=sanitized_analysis,
            global_api_contract_stub=global_api_contract_stub,
            **extra_kwargs)}]
        segment_meta = {
            "paper_content_len": len(paper_content_prompt),
            "overview_len": len(overview_prompt),
            "design_len": len(design_prompt),
            "task_len": len(task_prompt),
            "config_yaml_len": len(config_yaml),
            "code_files_len": len(code_files),
            "detailed_logic_analysis_len": len(sanitized_analysis),
            "global_api_contract_stub_len": len(global_api_contract_stub),
            "baseline_file_code_len": len(extra_kwargs.get("baseline_file_code", "")),
            "integration_hint_len": len(extra_kwargs.get("integration_hint", "")),
            "required_context_code_len": len(extra_kwargs.get("required_context_code", "")),
            "upstream_callers_code_len": len(extra_kwargs.get("upstream_callers_code", "")),
            "downstream_callees_code_len": len(extra_kwargs.get("downstream_callees_code", "")),
        }
        return write_msg, segment_meta

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
            detailed_logic_analysis_dict[clean_todo_file_name] = _sanitize_prompt_text(
                detailed_logic_analysis_response[0]['choices'][0]['message']['content'],
                max_chars=FEATURE_ANALYSIS_MAX_CHARS if prompt_set == "feature" else 12000,
            )
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
        if _violates_forbidden_python_file_name(clean_todo_file_name):
            existing_code, existing_source = _load_feature_file_code(clean_todo_file_name)
            if existing_code is None:
                raise PipelineError(
                    f"Planning produced a forbidden new file name: {clean_todo_file_name}. "
                    "Use an existing main-path file or a descriptive non-numbered file name."
                )
            logger.warning(
                f"[CODING] Allow existing file despite naming-rule match: {clean_todo_file_name} ({existing_source})"
            )

        save_todo_file_name_skip = _make_safe_artifact_stem(clean_todo_file_name)
        skip_path = f'{artifact_output_dir}/{save_todo_file_name_skip}_coding.txt'
        if os.path.exists(skip_path):
            repo_file_path = os.path.join(output_repo_dir, clean_todo_file_name)
            if os.path.exists(repo_file_path):
                with open(repo_file_path, 'r') as f:
                    done_file_dict[clean_todo_file_name] = f.read()
            done_file_lst.append(clean_todo_file_name)
            logger.info(f"  [SKIP] artifact already exists: {skip_path}")
            continue

        instruction_msg, segment_meta = get_write_msg(
            clean_todo_file_name,
            detailed_logic_analysis_dict.get(clean_todo_file_name, ""),
            done_file_lst,
        )
        trajectories.extend(instruction_msg)

        completion = None
        completion_json = None
        validated_code = ""
        last_error_kind = ""
        last_error_message = ""
        for attempt in range(3):
            try:
                safe_trajectories, payload_len = _prepare_messages_for_api(
                    trajectories,
                    clean_todo_file_name,
                    {
                        **segment_meta,
                        "attempt": attempt + 1,
                    },
                )
                completion = api_call(safe_trajectories)
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
                        "segment_meta": segment_meta,
                        "attempt": attempt + 1,
                        "payload_len": payload_len if 'payload_len' in locals() else None,
                    },
                )
                raise
            completion_json_try = json.loads(completion.model_dump_json())
            content_try = completion_json_try["choices"][0]["message"]["content"]
            code_try = extract_code_from_content(content_try) or content_try
            ok, error_kind, error_message = _validate_generated_code_attempt(
                clean_todo_file_name,
                code_try,
            )
            if ok:
                completion_json = completion_json_try
                validated_code = code_try
                break
            last_error_kind = error_kind
            last_error_message = error_message
            trajectories.append({"role": "assistant", "content": content_try})
            retry_message = _build_retry_message(error_kind, error_message)
            trajectories.append({
                "role": "user",
                "content": retry_message,
            })
            logger.warning(
                f"[CODING] Validation failed for {clean_todo_file_name} ({error_kind}), retry {attempt+1}/3: {error_message}"
            )

        if completion_json is None:
            raise PipelineError(
                f"Generated code for {clean_todo_file_name} failed validation after 3 attempts. "
                f"Last error [{last_error_kind}]: {last_error_message}"
            )

        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

        done_file_lst.append(clean_todo_file_name)

        os.makedirs(f'{output_repo_dir}', exist_ok=True)
        save_todo_file_name = _make_safe_artifact_stem(clean_todo_file_name)

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        code = validated_code
        if len(code) == 0:
            code = extract_code_from_content(message.content)
        if len(code) == 0:
            code = message.content

        with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w') as f:
            f.write(completion_json['choices'][0]['message']['content'])

        done_file_dict[clean_todo_file_name] = code
        new_definitions.extend(_extract_new_definitions(code))
        file_path = os.path.join(output_repo_dir, clean_todo_file_name)
        file_path = _ensure_path_within_root(output_repo_dir, file_path)
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
    parser.add_argument('--api_predefine_contract', type=str, default="")
    args = parser.parse_args()

    run_coding(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
        api_predefine_contract_path=args.api_predefine_contract or None,
    )
