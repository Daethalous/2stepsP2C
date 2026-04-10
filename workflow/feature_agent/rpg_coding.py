import ast
import copy
import json
import os
import re

from tqdm import tqdm

from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.exceptions import PipelineError
from core.llm_engine import chat_completion_with_retry, create_client
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.repo_index import collect_context_bundle, render_context_bundle_for_prompt
from core.utils import (
    build_code_interface_summary,
    contains_forbidden_placeholders,
    extract_code_from_content,
    format_paper_content_for_prompt,
    load_accumulated_cost,
    print_log_cost,
    print_response,
    read_python_files,
    save_accumulated_cost,
)
from workflow.baseline_agent.rpg_coding import (
    _extract_new_definitions,
    _validate_against_stub,
    format_signature_warnings,
    validate_cross_file_signatures,
)
from workflow.baseline_agent.rpg_interface_design import extract_stub_signatures
from workflow.coding import (
    BASELINE_INTERFACE_SUMMARY_MAX_CHARS,
    FEATURE_ANALYSIS_MAX_CHARS,
    FEATURE_API_CONTRACT_MAX_CHARS,
    FEATURE_CODE_FILES_MAX_CHARS,
    _build_done_code_context,
    _debug_log,
    _ensure_path_within_root,
    _load_baseline_interface_stub,
    _make_safe_artifact_stem,
    _prepare_messages_for_api,
    _prompt_path,
    _sanitize_payload,
    _sanitize_prompt_text,
)
from workflow.feature_agent.rpg_adapter import (
    build_feature_coding_context,
    get_feature_file_order,
    get_verified_stub_context,
    load_or_build_feature_rpg_bundle,
    load_verified_baseline_stubs,
)

logger = get_logger(__name__)


def run_rpg_coding(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    output_repo_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    prompt_set: str = None,
    baseline_repo_dir: str = None,
    live_repo_dir: str = None,
    baseline_interface_stub_path: str = None,
) -> None:
    _debug_log(
        run_id="initial",
        hypothesis_id="H1",
        location="workflow/feature_agent/rpg_coding.py:run_rpg_coding.entry",
        message="run_rpg_coding entry",
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
        format_paper_content_for_prompt(context_lst[0], max_chars=12000),
        max_chars=12000,
    ) if len(context_lst) > 0 else ""
    design_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[1], max_chars=16000),
        max_chars=16000,
    ) if len(context_lst) > 1 else ""
    task_prompt = _sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[2], max_chars=16000),
        max_chars=16000,
    ) if len(context_lst) > 2 else ""
    baseline_interface_stub = _load_baseline_interface_stub(
        output_dir, explicit_path=baseline_interface_stub_path
    )

    rpg, file_metadata = load_or_build_feature_rpg_bundle(
        output_dir=output_dir,
        baseline_interface_stub_path=baseline_interface_stub_path,
    )
    todo_file_lst = get_feature_file_order(rpg, ctx.todo_file_lst, file_metadata)
    done_file_lst = ["config.yaml"]
    done_file_dict = {}
    new_definitions = []

    stubs_dict = load_verified_baseline_stubs(baseline_interface_stub_path)
    has_stubs = len(stubs_dict) > 0
    if has_stubs:
        unique_stub_count = len(set(stubs_dict.values()))
        logger.info(f"[Feature RPG] Loaded {unique_stub_count} verified baseline stubs")
    else:
        logger.info("[Feature RPG] No verified baseline stubs available")

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

    def _get_feature_file_meta(path_name: str) -> dict:
        clean_name = sanitize_todo_file_name(path_name)
        return file_metadata.get(clean_name, {}) if clean_name else {}

    def _get_feature_context_bundle(path_name: str) -> dict:
        clean_name = sanitize_todo_file_name(path_name)
        return ctx.context_bundle_by_file.get(clean_name, {}) if clean_name else {}

    def _has_planned_public_interface_changes(path_name: str) -> bool:
        clean_name = sanitize_todo_file_name(path_name)
        return bool(feature_metadata.get("public_interface_changes_by_file", {}).get(clean_name, []))

    def _get_feature_stub_contract(path_name: str) -> dict:
        clean_name = sanitize_todo_file_name(path_name)
        return feature_metadata.get("stub_contracts_by_file", {}).get(clean_name, {}) if clean_name else {}

    def _is_constructor_contract_symbol(symbol_name: str) -> bool:
        owner_ref = _parse_hard_symbol_requirement(symbol_name)
        return bool(owner_ref.get("owner") and owner_ref.get("member") == "__init__")

    def _get_constructor_instantiation_changes(path_name: str) -> list[str]:
        clean_name = sanitize_todo_file_name(path_name)
        if not clean_name:
            return []
        return feature_metadata.get("constructor_instantiation_changes_by_file", {}).get(clean_name, [])

    def _has_planned_constructor_instantiation_changes(path_name: str, symbol_name: str = "") -> bool:
        change_details = _get_constructor_instantiation_changes(path_name)
        if not change_details:
            return False
        symbol_text = str(symbol_name or "").strip()
        if not symbol_text:
            return True
        owner_ref = _parse_hard_symbol_requirement(symbol_text)
        if owner_ref.get("owner") and owner_ref.get("member"):
            owner = owner_ref["owner"]
            member = owner_ref["member"]
            variants = {
                symbol_text,
                f"{owner}.{member}",
                f"{owner}::{member}",
            }
        else:
            variants = {symbol_text}
        for detail in change_details:
            detail_text = str(detail or "")
            if any(variant and variant in detail_text for variant in variants):
                return True
        return False

    def _is_entry_point_file(path_name: str) -> bool:
        clean_name = sanitize_todo_file_name(path_name)
        if clean_name in set(ctx.entrypoint_index.get("primary_files", [])):
            return True
        closure = _get_feature_closure(clean_name)
        for raw_entry in closure.get("entrypoints", []):
            entry_path = sanitize_todo_file_name(str(raw_entry).split("::", 1)[0].strip())
            if entry_path and entry_path == clean_name:
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
            if name == normalized or name == base_name:
                return True
        return False

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

        constructor_changes = _get_constructor_instantiation_changes(todo_file_name)
        if constructor_changes:
            hint_sections.append(
                "Constructor/instantiation changes touching this file:\n"
                + "\n".join(f"- {item}" for item in constructor_changes)
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

    def _normalize_expected_symbol(raw_symbol: str) -> str:
        requirement = _parse_hard_symbol_requirement(raw_symbol)
        if requirement.get("member"):
            return requirement["member"]
        return requirement.get("name", "")

    def _extract_python_defs(code_text: str) -> dict:
        classes = set(re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", code_text, flags=re.MULTILINE))
        functions = set(re.findall(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", code_text, flags=re.MULTILINE))
        return {"classes": classes, "functions": functions}

    def _parse_ast_facts(code_text: str) -> dict:
        facts = {
            "classes": set(),
            "functions": set(),
            "methods_by_class": {},
            "imported_names": set(),
            "imported_modules": set(),
            "called_names": set(),
            "assigned_names": set(),
        }
        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return facts
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                facts["classes"].add(node.name)
                methods = {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                facts["methods_by_class"][node.name] = methods
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                facts["functions"].add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[-1]
                    facts["imported_modules"].add(alias.name)
                    facts["imported_names"].add(alias.asname or module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    facts["imported_modules"].add(node.module)
                    facts["imported_names"].add(node.module.split(".")[-1])
                for alias in node.names:
                    facts["imported_names"].add(alias.asname or alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        facts["assigned_names"].add(target.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    facts["called_names"].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    facts["called_names"].add(node.func.attr)
        return facts

    def _get_feature_stub_conflicts(path_name: str, own_stub: str) -> list[str]:
        if not own_stub:
            return []
        contract = _get_feature_stub_contract(path_name)
        if not isinstance(contract, dict) or not contract:
            return []
        stub_sigs = extract_stub_signatures(own_stub)
        stub_facts = _parse_ast_facts(own_stub)
        conflicts = []
        for symbol_name in contract.get("required_top_level", []) or []:
            if symbol_name not in stub_facts["classes"] and symbol_name not in stub_facts["functions"]:
                conflicts.append(f"baseline stub missing top-level symbol `{symbol_name}`")
        for class_name, methods in (contract.get("required_methods", {}) or {}).items():
            if class_name not in stub_facts["classes"]:
                conflicts.append(f"baseline stub missing class `{class_name}`")
                continue
            available_methods = stub_facts["methods_by_class"].get(class_name, set())
            for method_name in methods:
                if method_name not in available_methods:
                    if method_name == "__init__" and _has_planned_constructor_instantiation_changes(
                        path_name,
                        f"{class_name}.{method_name}",
                    ):
                        continue
                    conflicts.append(f"baseline stub missing method `{class_name}.{method_name}`")
        for symbol_name, params in (contract.get("exact_params", {}) or {}).items():
            expected_params = [str(x).strip() for x in params if str(x).strip() not in {"self", "cls"}]
            stub_params = stub_sigs.get(symbol_name)
            if stub_params is None:
                if _is_constructor_contract_symbol(symbol_name) and _has_planned_constructor_instantiation_changes(
                    path_name,
                    symbol_name,
                ):
                    continue
                conflicts.append(f"baseline stub missing signature target `{symbol_name}`")
                continue
            actual_params = [str(x).strip() for x in stub_params if str(x).strip() not in {"self", "cls"}]
            if actual_params != expected_params:
                if _is_constructor_contract_symbol(symbol_name) and _has_planned_constructor_instantiation_changes(
                    path_name,
                    symbol_name,
                ):
                    continue
                conflicts.append(
                    f"baseline stub signature conflict for `{symbol_name}`: expected {expected_params}, got {actual_params}"
                )
        return conflicts

    def _should_treat_baseline_stub_as_reference(path_name: str, own_stub: str) -> bool:
        return (
            _has_planned_public_interface_changes(path_name)
            or _has_planned_constructor_instantiation_changes(path_name)
            or bool(
            _get_feature_stub_conflicts(path_name, own_stub)
            )
        )

    def _should_relax_baseline_signature_warnings(path_name: str, own_stub: str) -> bool:
        clean_name = sanitize_todo_file_name(path_name)
        return _should_treat_baseline_stub_as_reference(path_name, own_stub) or bool(
            feature_metadata.get("callsite_updates_by_file", {}).get(clean_name, [])
        )

    def _parse_hard_symbol_requirement(raw_symbol: str) -> dict:
        text = str(raw_symbol or "").strip()
        if not text:
            return {}
        lowered = text.lower()
        if any(marker in lowered for marker in (" and ", " or ")):
            return {}
        if any(marker in text for marker in ("/", "|", ",")):
            return {}
        signature_text = text.split("(", 1)[0].strip()
        if "." in signature_text:
            owner, member = signature_text.rsplit(".", 1)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", owner) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", member):
                return {"owner": owner, "member": member}
            return {}
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", signature_text):
            return {"name": signature_text}
        return {}

    def _split_target_symbol_requirements(todo_file_name: str) -> tuple[list[dict], list[str]]:
        closure = _get_feature_closure(todo_file_name)
        hard_requirements = []
        soft_hints = []
        seen_hard = set()
        seen_soft = set()
        for raw_symbol in closure.get("target_symbols", []):
            text = str(raw_symbol or "").strip()
            if not text:
                continue
            requirement = _parse_hard_symbol_requirement(text)
            if requirement:
                hard_key = (
                    requirement.get("owner", ""),
                    requirement.get("member", ""),
                    requirement.get("name", ""),
                )
                if hard_key not in seen_hard:
                    seen_hard.add(hard_key)
                    hard_requirements.append({"raw": text, **requirement})
            elif text not in seen_soft:
                seen_soft.add(text)
                soft_hints.append(text)
        return hard_requirements, soft_hints

    def _path_to_reference_token(raw_path: str) -> str:
        normalized = sanitize_todo_file_name(raw_path) or str(raw_path).strip().replace("\\", "/")
        base = os.path.splitext(os.path.basename(normalized))[0]
        if base == "__init__":
            base = os.path.basename(os.path.dirname(normalized))
        return base

    def _is_structured_reference_text(raw_text: str) -> bool:
        text = str(raw_text or "").strip()
        if not text:
            return False
        if "(" in text or ")" in text:
            base_text = text.split("(", 1)[0].strip()
            if base_text != text:
                if "::" in base_text or "/" in base_text or base_text.endswith(".py"):
                    return _is_structured_reference_text(base_text)
                return False
            return False
        if "::" in text:
            file_part, symbol_part = text.split("::", 1)
            return bool(_path_to_reference_token(file_part)) and bool(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", symbol_part.strip())
            )
        if any(ch.isspace() for ch in text) or "(" in text or ")" in text:
            return False
        if "/" in text or text.endswith(".py"):
            return bool(_path_to_reference_token(text))
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", text))

    def _extract_reference_tokens(raw_text: str) -> list[str]:
        text = str(raw_text or "").strip()
        if "(" in text or ")" in text:
            base_text = text.split("(", 1)[0].strip()
            if base_text != text and ("::" in base_text or "/" in base_text or base_text.endswith(".py")):
                text = base_text
        if not _is_structured_reference_text(text):
            return []
        tokens = []
        if "::" in text:
            file_part, symbol_part = text.split("::", 1)
            base = _path_to_reference_token(file_part)
            if base:
                tokens.append(base)
            symbol_tail = symbol_part.strip().split(".")[-1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol_tail):
                tokens.append(symbol_tail)
            return [item for item in dict.fromkeys(tokens) if item]
        if "/" in text or text.endswith(".py"):
            base = _path_to_reference_token(text)
            return [base] if base else []
        dotted_tokens = [part for part in text.split(".") if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part)]
        if dotted_tokens:
            tokens.append(dotted_tokens[-1])
            if len(dotted_tokens) == 1:
                tokens.append(dotted_tokens[0])
        return [item for item in dict.fromkeys(tokens) if item]

    def _get_structured_downstream_refs(todo_file_name: str) -> list[str]:
        clean_name = sanitize_todo_file_name(todo_file_name)
        references = []
        seen = set()
        for raw_ref in _get_feature_file_meta(clean_name).get("downstream_callees", []) or []:
            text = sanitize_todo_file_name(raw_ref) or str(raw_ref or "").strip()
            if text and text not in seen:
                seen.add(text)
                references.append(text)
        for raw_ref in _get_feature_context_bundle(clean_name).get("downstream_callee_files", []) or []:
            text = sanitize_todo_file_name(raw_ref) or str(raw_ref or "").strip()
            if text and text not in seen:
                seen.add(text)
                references.append(text)
        if references:
            return references
        for raw_ref in _get_feature_closure(clean_name).get("downstream_callees", []) or []:
            text = str(raw_ref or "").strip()
            if not _is_structured_reference_text(text):
                continue
            if text not in seen:
                seen.add(text)
                references.append(text)
        return references

    def _get_forward_wiring_tokens(todo_file_name: str) -> list[str]:
        tokens = []
        for raw_ref in _get_structured_downstream_refs(todo_file_name):
            tokens.extend(_extract_reference_tokens(raw_ref))
        return [item for item in dict.fromkeys(tokens) if item]

    def _build_wiring_debug_context(todo_file_name: str, code_text: str = "") -> dict:
        clean_name = sanitize_todo_file_name(todo_file_name)
        closure = _get_feature_closure(clean_name)
        file_meta = _get_feature_file_meta(clean_name)
        context_bundle = _get_feature_context_bundle(clean_name)
        downstream_refs = _get_structured_downstream_refs(clean_name)
        forward_tokens = _get_forward_wiring_tokens(clean_name)
        facts = _parse_ast_facts(code_text) if code_text else {
            "classes": set(),
            "functions": set(),
            "methods_by_class": {},
            "imported_names": set(),
            "imported_modules": set(),
            "called_names": set(),
            "assigned_names": set(),
        }
        evidence_pool = sorted(
            facts["imported_names"]
            | facts["called_names"]
            | facts["assigned_names"]
            | {item.split(".")[-1] for item in facts["imported_modules"]}
        )
        matched_tokens = [token for token in forward_tokens if token in evidence_pool]
        return {
            "file": clean_name,
            "is_caller_side": _is_caller_side_file(clean_name),
            "downstream_refs": downstream_refs,
            "forward_tokens": forward_tokens,
            "matched_tokens": matched_tokens,
            "evidence_pool": evidence_pool[:60],
            "closure_downstream_callees": list(closure.get("downstream_callees", []) or []),
            "file_metadata_downstream_callees": list(file_meta.get("downstream_callees", []) or []),
            "context_bundle_downstream_callee_files": list(context_bundle.get("downstream_callee_files", []) or []),
            "entrypoints": list(closure.get("entrypoints", []) or []),
            "focus_role_tags": list(closure.get("focus_role_tags", []) or []),
        }

    def _format_wiring_debug_prompt(debug_context: dict) -> str:
        if not isinstance(debug_context, dict):
            return "(no wiring debug context available)"
        lines = [
            f"Current file: {debug_context.get('file') or '(unknown)'}",
            f"Caller-side validation active: {debug_context.get('is_caller_side')}",
            "Planned downstream refs: " + (
                ", ".join(debug_context.get("downstream_refs", [])[:12]) or "(none)"
            ),
            "Expected wiring tokens: " + (
                ", ".join(debug_context.get("forward_tokens", [])[:12]) or "(none)"
            ),
            "Observed AST evidence tokens: " + (
                ", ".join(debug_context.get("evidence_pool", [])[:20]) or "(none)"
            ),
            "Matched tokens: " + (
                ", ".join(debug_context.get("matched_tokens", [])[:12]) or "(none)"
            ),
        ]
        return "\n".join(lines)

    def _is_caller_side_file(todo_file_name: str) -> bool:
        closure = _get_feature_closure(todo_file_name)
        role_tags = {
            str(tag).strip().lower()
            for tag in closure.get("focus_role_tags", [])
            if str(tag).strip()
        }
        return bool(
            _is_entry_point_file(todo_file_name)
            or any(tag in role_tags for tag in ("entrypoint", "main", "train", "eval", "registry", "factory"))
            or _get_forward_wiring_tokens(todo_file_name)
        )

    def _get_runtime_contract_error(todo_file_name: str, code_text: str) -> str:
        contract = feature_metadata.get("stub_contracts_by_file", {}).get(
            sanitize_todo_file_name(todo_file_name),
            {},
        )
        if not isinstance(contract, dict) or not contract:
            return ""
        facts = _parse_ast_facts(code_text)
        code_sigs = extract_stub_signatures(code_text)
        errors = []
        required_top_level = [item for item in contract.get("required_top_level", []) if str(item).strip()]
        for symbol_name in required_top_level:
            if symbol_name not in facts["classes"] and symbol_name not in facts["functions"]:
                errors.append(f"missing top-level symbol `{symbol_name}`")
        for class_name, methods in (contract.get("required_methods", {}) or {}).items():
            if class_name not in facts["classes"]:
                errors.append(f"missing class `{class_name}`")
                continue
            available_methods = facts["methods_by_class"].get(class_name, set())
            for method_name in methods:
                if method_name not in available_methods:
                    errors.append(f"missing method `{class_name}.{method_name}`")
        for symbol_name, params in (contract.get("exact_params", {}) or {}).items():
            expected_params = [str(x).strip() for x in params if str(x).strip() not in {"self", "cls"}]
            actual_params = code_sigs.get(symbol_name)
            if actual_params is None:
                errors.append(f"missing signature target `{symbol_name}`")
                continue
            if _is_constructor_contract_symbol(symbol_name):
                continue
            if len(actual_params) < len(expected_params) or actual_params[: len(expected_params)] != expected_params:
                errors.append(
                    f"parameter mismatch for `{symbol_name}`: expected {expected_params}, got {actual_params}"
                )
        if errors:
            return (
                f"Generated code for {todo_file_name} violates structured runtime contract: "
                + "; ".join(errors[:8])
            )
        return ""

    def _get_wiring_evidence_error(todo_file_name: str, code_text: str) -> str:
        if not _is_python_source_file(todo_file_name):
            return ""
        if not _is_caller_side_file(todo_file_name):
            return ""
        debug_context = _build_wiring_debug_context(todo_file_name, code_text)
        downstream_refs = debug_context["downstream_refs"]
        if not downstream_refs:
            return ""
        tokens = debug_context["forward_tokens"]
        if not tokens:
            return ""
        matched_tokens = debug_context["matched_tokens"]
        if tokens and not matched_tokens:
            return (
                f"Generated code for {todo_file_name} does not show AST-level wiring evidence for downstream "
                f"callee targets: {downstream_refs[:8]}. Expected wiring tokens: {tokens[:10]}. "
                f"Observed evidence tokens: {debug_context['evidence_pool'][:12]}"
            )
        return ""

    def _get_feature_validation_error(todo_file_name: str, code_text: str) -> tuple[str, str]:
        if not todo_file_name.endswith(".py"):
            return "", ""
        facts = _parse_ast_facts(code_text)
        allow_callsite_evidence = _is_caller_side_file(todo_file_name)
        hard_requirements, _ = _split_target_symbol_requirements(todo_file_name)
        missing_symbols = []
        for requirement in hard_requirements:
            if requirement.get("owner") and requirement.get("member"):
                owner = requirement["owner"]
                member = requirement["member"]
                matched = owner in facts["classes"] and member in facts["methods_by_class"].get(owner, set())
            else:
                normalized = requirement.get("name", "")
                matched = (
                    normalized in facts["functions"]
                    or normalized in facts["classes"]
                    or (
                        allow_callsite_evidence
                        and (
                            normalized in facts["imported_names"]
                            or normalized in facts["called_names"]
                            or normalized in facts["assigned_names"]
                        )
                    )
                )
            if not matched:
                missing_symbols.append(requirement["raw"])
        if missing_symbols:
            return "MissingTargetDefinition", (
                f"Generated code for {todo_file_name} does not contain expected target symbols: "
                f"{missing_symbols[:8]}"
            )
        wiring_error = _get_wiring_evidence_error(todo_file_name, code_text)
        if wiring_error:
            return "MissingWiringEvidence", wiring_error
        runtime_contract_error = _get_runtime_contract_error(todo_file_name, code_text)
        if runtime_contract_error:
            return "RuntimeContractDrift", runtime_contract_error
        return "", ""

    def _get_anti_simplification_constraints(todo_file_name: str) -> list[str]:
        closure = _get_feature_closure(todo_file_name)
        return [
            str(item).strip()
            for item in closure.get("interface_constraints", [])
            if str(item).strip().startswith("ANTI_SIMPLIFICATION:")
        ]

    def _iter_named_functions(tree: ast.AST, target_name: str) -> list[ast.AST]:
        matches = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                matches.append(node)
        return matches

    def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def _is_wrapper_like_statement(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Return):
            value = stmt.value
            return value is None or isinstance(value, (ast.Call, ast.Name, ast.Attribute, ast.Constant))
        if isinstance(stmt, ast.Expr):
            return isinstance(stmt.value, (ast.Call, ast.Name, ast.Attribute, ast.Constant))
        if isinstance(stmt, ast.Assign):
            return isinstance(stmt.value, (ast.Call, ast.Name, ast.Attribute, ast.Constant))
        if isinstance(stmt, ast.AnnAssign):
            return isinstance(stmt.value, (ast.Call, ast.Name, ast.Attribute, ast.Constant, type(None)))
        return False

    def _is_thin_wrapper_function(node: ast.AST) -> bool:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        body = _strip_docstring(list(node.body))
        if not body:
            return True
        if len(body) > 2:
            return False
        return all(_is_wrapper_like_statement(stmt) for stmt in body)

    def _build_reference_tokens(todo_file_name: str) -> list[str]:
        return _get_forward_wiring_tokens(todo_file_name)

    def _get_anti_simplification_error(todo_file_name: str, code_text: str) -> str:
        if not _is_python_source_file(todo_file_name):
            return ""
        anti_constraints = _get_anti_simplification_constraints(todo_file_name)
        if not anti_constraints:
            return ""

        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return ""

        hard_requirements, _ = _split_target_symbol_requirements(todo_file_name)
        normalized_targets = [
            requirement["name"]
            for requirement in hard_requirements
            if requirement.get("name")
        ]

        thin_targets = []
        for target_name in normalized_targets:
            matches = _iter_named_functions(tree, target_name)
            if matches and all(_is_thin_wrapper_function(node) for node in matches):
                thin_targets.append(target_name)
        if thin_targets:
            return (
                f"Generated code for {todo_file_name} over-simplifies required target symbols into thin wrappers: "
                f"{thin_targets[:6]}. Preserve the paper-specific implementation instead of a passthrough or bypass helper."
            )

        constraint_text = "\n".join(anti_constraints).lower()
        needs_real_path = bool(
            _is_caller_side_file(todo_file_name)
            or any(keyword in constraint_text for keyword in ("real execution path", "entrypoint", "callsite", "bypass", "parallel"))
        )
        if not needs_real_path:
            return ""

        facts = _parse_ast_facts(code_text)
        locally_defined = facts["classes"] | facts["functions"]
        evidence_pool = (
            facts["imported_names"]
            | facts["called_names"]
            | facts["assigned_names"]
            | {item.split(".")[-1] for item in facts["imported_modules"]}
        )
        downstream_refs = _get_structured_downstream_refs(todo_file_name)
        if not downstream_refs:
            return ""
        forward_targets = _build_reference_tokens(todo_file_name)
        external_hits = []
        for token in forward_targets:
            if token in locally_defined:
                continue
            if token in evidence_pool:
                external_hits.append(token)
        if forward_targets and not external_hits:
            return (
                f"Generated code for {todo_file_name} does not appear to reconnect the real execution path to any "
                f"downstream callee targets {downstream_refs[:6]}. "
                "Avoid parallel helpers and wire the actual caller/import/factory path to the paper logic."
            )
        return ""

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
            feature_error_kind, feature_error = _get_feature_validation_error(todo_file_name, code_text)
            if feature_error:
                return False, feature_error_kind, feature_error
            anti_simplification_error = _get_anti_simplification_error(todo_file_name, code_text)
            if anti_simplification_error:
                return False, "AntiSimplificationDrift", anti_simplification_error
        return True, "", ""

    def _build_retry_message(error_kind: str, error_message: str, debug_context: dict | None = None) -> str:
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
        if error_kind == "MissingTargetDefinition":
            return (
                "Output rejected [C001 MissingTargetDefinition]: expected target definitions or explicit wiring are missing.\n"
                f"Validation detail: {error_message}\n"
                "Add the required class/function/method definitions or the real import/call wiring. Return ONLY the complete corrected file."
            )
        if error_kind == "MissingWiringEvidence":
            wiring_debug = _format_wiring_debug_prompt(debug_context or {})
            return (
                "Output rejected [C002 MissingWiringEvidence]: the real execution path is not visibly wired back in AST-level imports/calls.\n"
                f"Validation detail: {error_message}\n"
                "You must fix this in the CURRENT file by adding AST-visible wiring evidence for at least one expected token.\n"
                "Accepted evidence includes explicit import names, constructor/function calls, registry/factory registrations, or assignments that reference the downstream target.\n"
                "Comments, strings, config-only mentions, or wiring moved into another helper file do NOT count.\n"
                f"Wiring debug context:\n{wiring_debug}\n"
                "Reconnect the true caller/factory/entrypoint path. Do not use a parallel helper path. Return ONLY the complete corrected file."
            )
        if error_kind == "RuntimeContractDrift":
            return (
                "Output rejected [C003 RuntimeContractDrift]: structured contract requirements are not satisfied.\n"
                f"Validation detail: {error_message}\n"
                "Satisfy the declared symbols, method set, and exact parameter contract. Return ONLY the fully corrected complete file."
            )
        if error_kind == "AntiSimplificationDrift":
            return (
                "Output rejected [C004 AntiSimplificationDrift]: the generated code over-simplifies a paper-critical path.\n"
                f"Validation detail: {error_message}\n"
                "You must preserve the real execution path, avoid thin wrappers or bypass helpers, "
                "and update the actual callers/imports/entrypoints that should reach this logic. "
                "Return ONLY the fully corrected complete file code."
            )
        return (
            "Output rejected: code still contains unfinished implementation markers "
            "(for example NotImplementedError/TODO/stub). "
            "Return ONLY the complete corrected file code, no explanations."
        )

    def _collect_contract_warnings(todo_file_name: str, code_text: str) -> list[str]:
        warnings = []
        runtime_contract_error = _get_runtime_contract_error(todo_file_name, code_text)
        if runtime_contract_error:
            warnings.append(f"RUNTIME CONTRACT: {runtime_contract_error}")
        return warnings

    def _build_final_repo_snapshot() -> dict[str, str]:
        snapshot = {}
        if output_repo_dir and os.path.isdir(output_repo_dir):
            snapshot.update(read_python_files(output_repo_dir))
        for path_name, code_text in done_file_dict.items():
            snapshot.setdefault(path_name, code_text)
        return snapshot

    def _run_final_contract_sweep() -> list[str]:
        errors = []
        final_snapshot = _build_final_repo_snapshot()
        for final_file in todo_file_lst:
            if final_file == "config.yaml":
                continue
            clean_file = sanitize_todo_file_name(final_file)
            code_text = final_snapshot.get(clean_file, "")
            if not code_text:
                errors.append(f"{clean_file}: missing generated code in final snapshot.")
                continue
            ok, error_kind, error_message = _validate_generated_code_attempt(clean_file, code_text)
            if not ok:
                errors.append(f"{clean_file}: [{error_kind}] {error_message}")
            if _is_python_source_file(clean_file):
                own_stub = ""
                if has_stubs:
                    own_stub, _ = get_verified_stub_context(stubs_dict, clean_file, rpg)
                signature_warnings = validate_cross_file_signatures(code_text, clean_file, final_snapshot, rpg)
                if signature_warnings and not _should_relax_baseline_signature_warnings(clean_file, own_stub):
                    errors.extend(f"{clean_file}: {warning}" for warning in signature_warnings)
                errors.extend(
                    f"{clean_file}: {warning}"
                    for warning in _collect_contract_warnings(clean_file, code_text)
                )
                if has_stubs:
                    stub_warnings = _validate_against_stub(code_text, clean_file, own_stub)
                    if stub_warnings and not _should_treat_baseline_stub_as_reference(clean_file, own_stub):
                        errors.extend(f"{clean_file}: {warning}" for warning in stub_warnings)
        return errors

    def _find_analysis_response_path(todo_file_name: str) -> str | None:
        candidates = _resolve_todo_candidates(todo_file_name)
        for candidate in candidates:
            safe_name = _make_safe_artifact_stem(candidate)
            path = os.path.join(output_dir, f"{safe_name}_simple_analysis_response.json")
            legacy_path = os.path.join(output_dir, f"{candidate.replace('/', '_')}_simple_analysis_response.json")
            if os.path.exists(path):
                return path
            if os.path.exists(legacy_path):
                return legacy_path
        return None

    if prompt_set == "feature":
        existing = read_python_files(output_repo_dir)
        for fname, content in existing.items():
            if fname not in done_file_dict:
                done_file_dict[fname] = content

    code_msg = [
        {
            "role": "system",
            "content": render_prompt(_prompt_path(prompt_set, "coding_system.txt"), paper_format=paper_format),
        }
    ]
    code_msg = _sanitize_payload(code_msg)

    def get_write_msg(todo_file_name: str, detailed_logic_analysis: str, done_file_lst: list):
        if prompt_set == "feature":
            rpg_context = build_feature_coding_context(
                rpg,
                todo_file_name,
                done_file_dict,
                max_total_chars=FEATURE_CODE_FILES_MAX_CHARS,
            )
            context_bundle = _build_context_bundle_for_file(todo_file_name)
            bundle_prompt = render_context_bundle_for_prompt(context_bundle) if context_bundle else {}
            code_sections = []
            if rpg_context and rpg_context != "(no dependency context available)":
                code_sections.append(rpg_context)
            if bundle_prompt:
                shared_code = bundle_prompt.get("shared_interfaces_code", "(none)")
                if shared_code and shared_code != "(none)":
                    code_sections.append(shared_code)
            if not code_sections:
                code_sections.append(
                    _build_done_code_context(
                        done_file_dict,
                        done_file_lst,
                        max_total_chars=FEATURE_CODE_FILES_MAX_CHARS,
                    )
                )
            code_files = _sanitize_prompt_text(
                "\n\n".join(section for section in code_sections if section),
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
        contract_text = baseline_interface_stub
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
                extra_kwargs["runtime_contract_summary"] = bundle_prompt.get("runtime_contract_summary", "(none)")
                extra_kwargs["runtime_contract_checks"] = bundle_prompt.get("runtime_contract_checks", "(none)")
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {bundle_prompt.get('focus_file_source', 'new')}\n{focus_file_code}"
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
                    "runtime_contract_summary",
                    "runtime_contract_checks",
                ):
                    extra_kwargs.setdefault(key, "(none)")

            closure = _get_feature_closure(todo_file_name)
            own_stub = ""
            dep_stubs = ""
            if has_stubs and _is_python_source_file(todo_file_name):
                own_stub, dep_stubs = get_verified_stub_context(stubs_dict, todo_file_name, rpg)
            if own_stub or dep_stubs:
                contract_parts = []
                stub_conflicts = _get_feature_stub_conflicts(todo_file_name, own_stub)
                if own_stub:
                    if _should_treat_baseline_stub_as_reference(todo_file_name, own_stub):
                        conflict_note = ""
                        if stub_conflicts:
                            conflict_note = "\nFeature contract overrides conflicting baseline clauses:\n" + "\n".join(
                                f"- {item}" for item in stub_conflicts[:8]
                            )
                        contract_parts.append(
                            "## Baseline Stub Reference\n"
                            "This baseline stub is a compatibility reference. "
                            "If the planned feature requires public API changes, keep synchronized edits consistent.\n"
                            f"{conflict_note}\n"
                            f"```python\n{own_stub}\n```"
                        )
                    else:
                        contract_parts.append(
                            "## Verified Baseline Interface Contract\n"
                            "You MUST preserve these signatures unless the prompt explicitly requires a synchronized public API change.\n"
                            f"```python\n{own_stub}\n```"
                        )
                if dep_stubs:
                    contract_parts.append(
                        "## Dependency Interfaces\n"
                        "Use these verified dependency signatures exactly when calling into baseline modules.\n"
                        f"{dep_stubs}"
                    )
                contract_parts.append("## Baseline Interface Stub Summary\n" + baseline_interface_stub)
                contract_text = _sanitize_prompt_text(
                    "\n\n".join(contract_parts),
                    max_chars=FEATURE_API_CONTRACT_MAX_CHARS,
                )
            extra_kwargs["integration_hint"] = _sanitize_prompt_text(
                extra_kwargs["integration_hint"],
                max_chars=2600,
            )

        sanitized_analysis = _sanitize_prompt_text(
            detailed_logic_analysis,
            max_chars=FEATURE_ANALYSIS_MAX_CHARS if prompt_set == "feature" else 12000,
        )

        write_msg = [
            {
                "role": "user",
                "content": render_prompt(
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
                    global_api_contract_stub=contract_text,
                    baseline_interface_stub=contract_text,
                    **extra_kwargs,
                ),
            }
        ]
        segment_meta = {
            "paper_content_len": len(paper_content_prompt),
            "overview_len": len(overview_prompt),
            "design_len": len(design_prompt),
            "task_len": len(task_prompt),
            "config_yaml_len": len(config_yaml),
            "code_files_len": len(code_files),
            "detailed_logic_analysis_len": len(sanitized_analysis),
            "baseline_interface_stub_len": len(contract_text),
            "baseline_file_code_len": len(extra_kwargs.get("baseline_file_code", "")),
            "integration_hint_len": len(extra_kwargs.get("integration_hint", "")),
            "required_context_code_len": len(extra_kwargs.get("required_context_code", "")),
            "upstream_callers_code_len": len(extra_kwargs.get("upstream_callers_code", "")),
            "downstream_callees_code_len": len(extra_kwargs.get("downstream_callees_code", "")),
            "runtime_contract_summary_len": len(extra_kwargs.get("runtime_contract_summary", "")),
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
                detailed_logic_analysis_response[0]["choices"][0]["message"]["content"],
                max_chars=FEATURE_ANALYSIS_MAX_CHARS if prompt_set == "feature" else 12000,
            )
        else:
            logger.warning(f"[WARNING] Analysis file not found for {todo_file_name}. Continue with empty analysis.")
            detailed_logic_analysis_dict[clean_todo_file_name] = ""

    artifact_output_dir = os.path.join(output_dir, "coding_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)
    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    signature_report = []

    def _persist_coding_debug_state(
        clean_todo_file_name: str,
        trajectories_payload: list,
        responses_payload: list,
        *,
        code_text: str = "",
        attempt: int | None = None,
        error_kind: str = "",
        error_message: str = "",
        segment_meta: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        safe_name = _make_safe_artifact_stem(clean_todo_file_name)
        debug_context = {
            "file": clean_todo_file_name,
            "attempt": attempt,
            "error_kind": error_kind,
            "error_message": error_message,
            "segment_meta": segment_meta or {},
            "wiring_debug": _build_wiring_debug_context(clean_todo_file_name, code_text),
        }
        if extra:
            debug_context.update(extra)
        with open(
            os.path.join(artifact_output_dir, f"{safe_name}_coding_trajectories.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(trajectories_payload, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(artifact_output_dir, f"{safe_name}_coding_responses.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(responses_payload, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(artifact_output_dir, f"{safe_name}_coding_debug_context.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(debug_context, f, ensure_ascii=False, indent=2)

    for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
        responses = []
        trajectories = copy.deepcopy(code_msg)

        current_stage = f"[FEATURE_RPG_CODING] {todo_file_name}"
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
        skip_path = os.path.join(artifact_output_dir, f"{save_todo_file_name_skip}_coding.txt")
        if os.path.exists(skip_path):
            repo_file_path = os.path.join(output_repo_dir, clean_todo_file_name)
            if os.path.exists(repo_file_path):
                with open(repo_file_path, "r", encoding="utf-8") as f:
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
                    {**segment_meta, "attempt": attempt + 1},
                )
                completion = api_call(safe_trajectories)
            except Exception as api_exc:
                _debug_log(
                    run_id="initial",
                    hypothesis_id="H4",
                    location="workflow/feature_agent/rpg_coding.py:api_call",
                    message="api call failed",
                    data={
                        "todo_file_name": clean_todo_file_name,
                        "error_type": type(api_exc).__name__,
                        "error": str(api_exc),
                        "status_code": getattr(api_exc, "status_code", None),
                        "segment_meta": segment_meta,
                        "attempt": attempt + 1,
                        "payload_len": payload_len if "payload_len" in locals() else None,
                    },
                )
                _persist_coding_debug_state(
                    clean_todo_file_name,
                    trajectories,
                    responses,
                    attempt=attempt + 1,
                    error_kind="api_call_exception",
                    error_message=str(api_exc),
                    segment_meta=segment_meta,
                    extra={"prepared_payload_len": payload_len if "payload_len" in locals() else None},
                )
                raise

            completion_json_try = json.loads(completion.model_dump_json())
            responses.append(completion_json_try)
            content_try = completion_json_try["choices"][0]["message"]["content"]
            code_try = extract_code_from_content(content_try) or content_try
            ok, error_kind, error_message = _validate_generated_code_attempt(clean_todo_file_name, code_try)
            if not ok:
                last_error_kind = error_kind
                last_error_message = error_message
                trajectories.append({"role": "assistant", "content": content_try})
                retry_message = _build_retry_message(
                    error_kind,
                    error_message,
                    debug_context=_build_wiring_debug_context(clean_todo_file_name, code_try)
                    if error_kind == "MissingWiringEvidence"
                    else None,
                )
                trajectories.append({"role": "user", "content": retry_message})
                logger.warning(
                    f"[CODING] Validation failed for {clean_todo_file_name} ({error_kind}), retry {attempt+1}/3: {error_message}"
                )
                _persist_coding_debug_state(
                    clean_todo_file_name,
                    trajectories,
                    responses,
                    code_text=code_try,
                    attempt=attempt + 1,
                    error_kind=error_kind,
                    error_message=error_message,
                    segment_meta=segment_meta,
                    extra={"retry_message": retry_message},
                )
                continue

            blocking_warnings = []
            advisory_warnings = []
            if _is_python_source_file(clean_todo_file_name):
                own_stub = ""
                if has_stubs:
                    own_stub, _ = get_verified_stub_context(stubs_dict, clean_todo_file_name, rpg)
                signature_warnings = validate_cross_file_signatures(code_try, clean_todo_file_name, done_file_dict, rpg)
                if signature_warnings:
                    if _should_relax_baseline_signature_warnings(clean_todo_file_name, own_stub):
                        advisory_warnings.extend([f"FEATURE-FIRST SIGNATURE REFERENCE: {item}" for item in signature_warnings])
                    else:
                        blocking_warnings.extend(signature_warnings)
                blocking_warnings.extend(
                    _collect_contract_warnings(clean_todo_file_name, code_try)
                )
                if has_stubs:
                    stub_warnings = _validate_against_stub(code_try, clean_todo_file_name, own_stub)
                    if _should_treat_baseline_stub_as_reference(clean_todo_file_name, own_stub):
                        advisory_warnings.extend([f"BASELINE STUB REFERENCE: {item}" for item in stub_warnings])
                    else:
                        blocking_warnings.extend(stub_warnings)

            all_warnings = blocking_warnings + advisory_warnings
            if blocking_warnings and attempt < 2:
                warning_text = format_signature_warnings(all_warnings)
                trajectories.append({"role": "assistant", "content": content_try})
                trajectories.append(
                    {
                        "role": "user",
                        "content": (
                            f"Potential issues detected in your code:\n{warning_text}\n"
                            "Please fix any function call mismatches or contract drift while keeping synchronized edits consistent. "
                            "Return the complete corrected code."
                        ),
                    }
                )
                logger.warning(
                    f"[CODING] RPG validation retry for {clean_todo_file_name} {attempt+1}/3: {all_warnings}"
                )
                for warning in all_warnings:
                    signature_report.append(f"{clean_todo_file_name}: {warning}")
                continue

            if all_warnings:
                for warning in all_warnings:
                    logger.warning(f"[FEATURE RPG WARNING] {warning}")
                    signature_report.append(f"{clean_todo_file_name}: {warning} (accepted)")

            completion_json = completion_json_try
            validated_code = code_try
            break

        if completion_json is None:
            _persist_coding_debug_state(
                clean_todo_file_name,
                trajectories,
                responses,
                attempt=3,
                error_kind=last_error_kind,
                error_message=last_error_message,
                segment_meta=segment_meta,
            )
            raise PipelineError(
                f"Generated code for {clean_todo_file_name} failed validation after 3 attempts. "
                f"Last error [{last_error_kind}]: {last_error_message}"
            )

        message = completion.choices[0].message
        trajectories.append({"role": message.role, "content": message.content})
        done_file_lst.append(clean_todo_file_name)

        os.makedirs(output_repo_dir, exist_ok=True)
        save_todo_file_name = _make_safe_artifact_stem(clean_todo_file_name)
        print_response(completion_json)
        total_accumulated_cost = print_log_cost(
            completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost
        )

        code = validated_code or extract_code_from_content(message.content) or message.content
        with open(os.path.join(artifact_output_dir, f"{save_todo_file_name}_coding.txt"), "w", encoding="utf-8") as f:
            f.write(completion_json["choices"][0]["message"]["content"])

        done_file_dict[clean_todo_file_name] = code
        new_definitions.extend(_extract_new_definitions(code))
        file_path = os.path.join(output_repo_dir, clean_todo_file_name)
        file_path = _ensure_path_within_root(output_repo_dir, file_path)
        if os.path.isdir(file_path):
            logger.warning(f"[CODING] Skip writing because target path is a directory: {file_path}")
            continue
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

    final_sweep_errors = _run_final_contract_sweep()
    if final_sweep_errors:
        with open(
            os.path.join(artifact_output_dir, "feature_rpg_final_sweep_debug.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "errors": final_sweep_errors,
                    "todo_file_lst": todo_file_lst,
                    "generated_files": sorted(done_file_dict.keys()),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        report_path = os.path.join(artifact_output_dir, "feature_rpg_signature_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Feature RPG Final Contract Sweep\n")
            f.write("=" * 50 + "\n\n")
            for entry in final_sweep_errors:
                f.write(f"{entry}\n")
        raise PipelineError(
            "Feature RPG final closure sweep failed: " + " | ".join(final_sweep_errors[:12])
        )

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
    if signature_report:
        report_path = os.path.join(artifact_output_dir, "feature_rpg_signature_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Feature RPG Signature Validation Report\n")
            f.write("=" * 50 + "\n\n")
            for entry in signature_report:
                f.write(f"  {entry}\n")

