"""
RPG Interface Design — Per-file, topologically-ordered stub generation.

Replaces the old rpg_api_predefine.py with a strict, per-file approach:
  1. Iterates files in RPG topological order
  2. For each file, shows the LLM the VERIFIED stubs of its dependencies
  3. LLM outputs a signature-only stub (imports + class/def + pass)
  4. Validates with ast.parse() + structural checks
  5. Saves to stubs/ directory for downstream use

Usage:
    Called from rpg_pipeline.py as the "interface_design" stage.
    Can also run standalone:
        python -m workflow.baseline_agent.rpg_interface_design \\
            --output_dir outputs/iTransformer_artifacts_rpg2
"""

import ast
import copy
import json
import keyword
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

from core.data_loader import (
    load_paper_content,
    load_pipeline_context,
    sanitize_todo_file_name,
)
from core.llm_engine import chat_completion_with_retry, create_client
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.utils import (
    extract_code_from_content,
    format_paper_content_for_prompt,
    load_accumulated_cost,
    print_log_cost,
    print_response,
    save_accumulated_cost,
)
from workflow.baseline_agent.build_rpg import PipelineRPG, build_rpg_from_planning

logger = get_logger(__name__)

FALLBACK_MARKER = "auto-generated fallback"
STUB_STATUS_VERIFIED = "verified"
STUB_STATUS_FAILED = "failed"
STUB_STATUS_SKIPPED = "skipped"
STUB_INDEX_VERSION = 2
LIGHT_REVIEW_FILES = {"config.py", "main.py", "__init__.py"}
STRICT_CONTRACT_FILES = {
    "dataset_loader.py",
    "evaluation.py",
    "evaluator.py",
    "metrics.py",
    "model.py",
    "trainer.py",
    "utils.py",
}
CONTRACT_TEXT_STOPWORDS = {
    "and",
    "class",
    "classes",
    "export",
    "exports",
    "function",
    "functions",
    "method",
    "methods",
    "or",
    "the",
    "with",
}


# ---------------------------------------------------------------------------
# Stub validation
# ---------------------------------------------------------------------------

def _validate_stub_syntax(code: str) -> Tuple[bool, str]:
    """Validate that the stub is parseable Python."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}"


def _validate_stub_structure(code: str) -> Tuple[bool, str]:
    """
    Validate structural requirements for a stub:
    - Must contain at least one class or function definition
    - Function bodies must be `pass` only (no logic)
    - No NotImplementedError or TODO
    """
    # Must have at least one def or class
    if "def " not in code and "class " not in code:
        return False, "Stub must contain at least one 'class' or 'def' definition."

    # Check for disallowed tokens
    low = code.lower()
    if "notimplementederror" in low:
        return False, "Stub must not contain 'NotImplementedError'. Use 'pass' for bodies."
    if "# todo" in low or "todo:" in low:
        return False, "Stub must not contain TODO comments."

    # Check that function bodies are pass-only (heuristic)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "Cannot parse stub for structural validation."

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A valid stub body should be:
            #   - Just `pass`
            #   - Or a docstring followed by `pass`
            #   - Or just a docstring (we allow this too)
            body = node.body
            non_docstring = []
            for stmt in body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                    continue  # docstring
                non_docstring.append(stmt)

            if len(non_docstring) == 0:
                # Only docstring, OK (implicit return None)
                continue
            if len(non_docstring) == 1 and isinstance(non_docstring[0], ast.Pass):
                continue
            # If there's an Ellipsis (...), also OK
            if (len(non_docstring) == 1
                    and isinstance(non_docstring[0], ast.Expr)
                    and isinstance(getattr(non_docstring[0], 'value', None), ast.Constant)
                    and getattr(non_docstring[0].value, 'value', None) is ...):
                continue

            # Check for logic statements
            for stmt in non_docstring:
                if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With,
                                     ast.Try, ast.Assign, ast.AugAssign,
                                     ast.Return, ast.Raise)):
                    return False, (
                        f"Function '{node.name}' contains implementation logic "
                        f"({type(stmt).__name__} at line {stmt.lineno}). "
                        f"Stub bodies must be 'pass' only."
                    )

    return True, ""


def validate_stub(code: str) -> Tuple[bool, str]:
    """Run all stub validations. Returns (ok, error_message)."""
    ok, err = _validate_stub_syntax(code)
    if not ok:
        return False, err
    ok, err = _validate_stub_structure(code)
    if not ok:
        return False, err
    return True, ""


def _is_fallback_stub(code: str) -> bool:
    return FALLBACK_MARKER in (code or "").lower()


# ---------------------------------------------------------------------------
# Stub extraction helpers
# ---------------------------------------------------------------------------

def extract_stub_signatures(code: str) -> Dict[str, List[str]]:
    """
    Extract function/class signatures from a stub for comparison.

    Returns:
        Dict mapping "function_name" -> [param_names]
        For methods: "ClassName.method_name" -> [param_names]
    """
    signatures = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return signatures

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in node.args.args:
                if arg.arg not in ("self", "cls"):
                    params.append(arg.arg)
            if node.args.vararg:
                params.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                params.append(f"**{node.args.kwarg.arg}")
            signatures[node.name] = params

        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = []
                    for arg in item.args.args:
                        if arg.arg not in ("self", "cls"):
                            params.append(arg.arg)
                    if item.args.vararg:
                        params.append(f"*{item.args.vararg.arg}")
                    if item.args.kwarg:
                        params.append(f"**{item.args.kwarg.arg}")
                    signatures[f"{node.name}.{item.name}"] = params

    return signatures


def _normalize_stub_text(raw: str) -> str:
    """Clean up LLM output to extract the stub code."""
    if not isinstance(raw, str):
        return ""
    txt = raw.strip()
    # Remove markdown code fences
    if txt.startswith("```python"):
        txt = txt[len("```python"):].strip()
    elif txt.startswith("```"):
        txt = txt[3:].strip()
    if txt.endswith("```"):
        txt = txt[:-3].strip()
    # Remove leading "python\n" if present
    if txt.lower().startswith("python\n"):
        txt = txt.split("\n", 1)[1]
    return txt.strip()


def _extract_top_level_defs(code: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Extract top-level defs/classes and class methods from Python code."""
    top_level: Set[str] = set()
    methods_by_class: Dict[str, Set[str]] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return top_level, methods_by_class

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level.add(node.name)
        if isinstance(node, ast.ClassDef):
            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            methods_by_class[node.name] = methods
    return top_level, methods_by_class


def _extract_signature_param_names(signature_text: str) -> List[str]:
    params_text = signature_text.strip()
    if not params_text:
        return []
    params: List[str] = []
    for raw_part in params_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        name = part.split(":", 1)[0].strip()
        name = name.split("=", 1)[0].strip()
        if name in {"self", "cls"}:
            continue
        params.append(name)
    return params


def _extract_named_signatures(text: str) -> Dict[str, List[str]]:
    signatures: Dict[str, List[str]] = {}
    for name, params in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)", text or ""):
        if name in {
            "shape", "dtype", "dict", "list", "tuple", "set", "Optional", "Tuple", "List", "Dict",
            "Callable", "Sequence", "Union", "Mapping",
        }:
            continue
        if name[0].isdigit():
            continue
        signatures[name] = _extract_signature_param_names(params)
    return signatures


def _extract_expected_function_names(text: str) -> Set[str]:
    names = set()
    for name in _extract_named_signatures(text).keys():
        if name.startswith("_") or name[0].islower():
            names.add(name)
    return names


def _empty_stub_contract() -> Dict[str, Any]:
    return {
        "required_top_level": set(),
        "required_methods": {},
        "exact_params": {},
        "disallow_extra_public_top_level": False,
        "disallow_extra_public_methods": False,
    }


def _normalize_stub_contract(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _empty_stub_contract()
    if not isinstance(contract, dict):
        return normalized
    normalized["required_top_level"] = {
        str(name).strip()
        for name in (contract.get("required_top_level") or set())
        if str(name).strip() and not keyword.iskeyword(str(name).strip())
    }
    required_methods: Dict[str, Set[str]] = {}
    for class_name, method_names in (contract.get("required_methods") or {}).items():
        class_key = str(class_name).strip()
        if not class_key:
            continue
        required_methods[class_key] = {
            str(name).strip()
            for name in (method_names or set())
            if str(name).strip() and not keyword.iskeyword(str(name).strip())
        }
    normalized["required_methods"] = required_methods
    exact_params: Dict[str, List[str]] = {}
    for symbol_name, params in (contract.get("exact_params") or {}).items():
        symbol_key = str(symbol_name).strip()
        if not symbol_key:
            continue
        exact_params[symbol_key] = [str(param).strip() for param in (params or []) if str(param).strip()]
    normalized["exact_params"] = exact_params
    normalized["disallow_extra_public_top_level"] = bool(contract.get("disallow_extra_public_top_level"))
    normalized["disallow_extra_public_methods"] = bool(contract.get("disallow_extra_public_methods"))
    return normalized


def _contract_has_requirements(contract: Optional[Dict[str, Any]]) -> bool:
    normalized = _normalize_stub_contract(contract)
    return bool(
        normalized["required_top_level"]
        or normalized["required_methods"]
        or normalized["exact_params"]
    )


def _merge_stub_contracts(
    primary: Optional[Dict[str, Any]],
    secondary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = _normalize_stub_contract(primary)
    extra = _normalize_stub_contract(secondary)
    merged["required_top_level"].update(extra["required_top_level"])
    for class_name, method_names in extra["required_methods"].items():
        merged["required_methods"].setdefault(class_name, set()).update(method_names)
    for symbol_name, params in extra["exact_params"].items():
        if symbol_name not in merged["exact_params"] or not merged["exact_params"][symbol_name]:
            merged["exact_params"][symbol_name] = list(params)
    merged["disallow_extra_public_top_level"] = (
        merged["disallow_extra_public_top_level"] or extra["disallow_extra_public_top_level"]
    )
    merged["disallow_extra_public_methods"] = (
        merged["disallow_extra_public_methods"] or extra["disallow_extra_public_methods"]
    )
    return merged


def _legacy_stub_contract(target_file: str, file_analysis: str) -> Dict[str, Any]:
    base = os.path.basename(target_file)
    contract = _empty_stub_contract()

    if base == "model.py":
        contract["required_top_level"] = {"TransformerBaseline"}
        contract["required_methods"] = {"TransformerBaseline": {"__init__", "forward"}}
        contract["exact_params"] = {
            "TransformerBaseline.__init__": [
                "T", "S", "N", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
            ],
            "TransformerBaseline.forward": ["x"],
        }
        contract["disallow_extra_public_top_level"] = True
        contract["disallow_extra_public_methods"] = True
        return contract

    if base == "trainer.py":
        contract["required_top_level"] = {"Trainer"}
        contract["required_methods"] = {
            "Trainer": {"__init__", "train", "_train_epoch", "_validate", "save_checkpoint", "load_checkpoint"}
        }
        contract["exact_params"] = {
            "Trainer.__init__": ["model", "optimizer", "criterion", "device", "out_dir", "scaler"],
            "Trainer.train": ["train_loader", "val_loader", "epochs", "save_best", "early_stop_patience"],
            "Trainer._train_epoch": ["train_loader"],
            "Trainer._validate": ["val_loader"],
            "Trainer.save_checkpoint": ["path"],
            "Trainer.load_checkpoint": ["path"],
        }
        contract["disallow_extra_public_top_level"] = True
        contract["disallow_extra_public_methods"] = True
        return contract

    if base in {"evaluation.py", "evaluator.py"}:
        class_name = "Evaluator"
        contract["required_top_level"] = {class_name}
        contract["required_methods"] = {class_name: {"__init__", "evaluate", "predict_batch"}}
        contract["exact_params"] = {
            f"{class_name}.__init__": ["model", "scaler", "device"],
            f"{class_name}.evaluate": ["test_loader"],
            f"{class_name}.predict_batch": ["x"],
        }
        contract["disallow_extra_public_top_level"] = True
        contract["disallow_extra_public_methods"] = True
        return contract

    if base == "metrics.py":
        contract["required_top_level"] = {"mse", "mae"}
        contract["exact_params"] = {
            "mse": ["y_true", "y_pred"],
            "mae": ["y_true", "y_pred"],
        }
        contract["disallow_extra_public_top_level"] = True
        return contract

    if base == "dataset_loader.py":
        contract["required_top_level"] = {
            "load_csv", "compute_scaler", "TimeSeriesDataset", "create_dataloaders",
        }
        contract["required_methods"] = {"TimeSeriesDataset": {"__init__", "__len__", "__getitem__"}}
        contract["exact_params"] = {
            "load_csv": ["path", "ignore_timestamp"],
            "compute_scaler": ["train_array"],
            "TimeSeriesDataset.__init__": ["data", "T", "S", "start", "end", "mean", "std"],
            "TimeSeriesDataset.__getitem__": ["idx"],
            "create_dataloaders": ["array", "T", "S", "split_bounds", "batch_size", "mean", "std", "num_workers"],
        }
        contract["disallow_extra_public_top_level"] = True
        return contract

    if base == "utils.py":
        function_names = _extract_expected_function_names(file_analysis)
        if function_names:
            contract["required_top_level"] = function_names
            contract["disallow_extra_public_top_level"] = True
            exact_params = {}
            for func_name, params in _extract_named_signatures(file_analysis).items():
                if func_name in function_names:
                    exact_params[func_name] = params
            contract["exact_params"] = exact_params
        return contract

    class_matches = set(re.findall(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", file_analysis or ""))
    if class_matches:
        contract["required_top_level"] = class_matches
    return contract


def _flatten_string_fragments(value: Any) -> List[str]:
    fragments: List[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            fragments.append(cleaned)
        return fragments
    if isinstance(value, dict):
        for item in value.values():
            fragments.extend(_flatten_string_fragments(item))
        return fragments
    if isinstance(value, (list, tuple, set)):
        for item in value:
            fragments.extend(_flatten_string_fragments(item))
    return fragments


def _extract_dotted_signatures(text: str) -> Dict[str, List[str]]:
    signatures: Dict[str, List[str]] = {}
    for name, params in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\(([^()]*)\)",
        text or "",
    ):
        root_name = name.split(".", 1)[0]
        if root_name in {
            "shape", "dtype", "dict", "list", "tuple", "set", "Optional", "Tuple", "List", "Dict",
            "Callable", "Sequence", "Union", "Mapping",
        }:
            continue
        if name.startswith(".") or name[0].isdigit():
            continue
        signatures[name] = _extract_signature_param_names(params)
    return signatures


def _extract_class_method_refs(text: str) -> List[Tuple[str, str]]:
    refs = []
    for class_name, method_name in re.findall(
        r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        text or "",
    ):
        refs.append((class_name, method_name))
    return refs


def _extract_exported_symbol_names(text: str) -> List[str]:
    match = re.search(r"\bExports?\s+([^.:\n]+)", text or "", flags=re.IGNORECASE)
    if not match:
        return []
    head = re.split(r"\bwith\b|\bmust\b|:", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
    names: List[str] = []
    for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", head):
        lowered = ident.lower()
        if lowered in CONTRACT_TEXT_STOPWORDS or keyword.iskeyword(ident):
            continue
        if ident not in names:
            names.append(ident)
    return names


def _register_required_method(contract: Dict[str, Any], class_name: str, method_name: str) -> None:
    class_name = str(class_name or "").strip()
    method_name = str(method_name or "").strip()
    if not class_name or not method_name:
        return
    contract["required_top_level"].add(class_name)
    contract["required_methods"].setdefault(class_name, set()).add(method_name)


def _register_signature(contract: Dict[str, Any], symbol_name: str, params: List[str]) -> None:
    symbol_name = str(symbol_name or "").strip()
    if not symbol_name:
        return
    if keyword.iskeyword(symbol_name):
        return
    clean_params = [str(param).strip() for param in (params or []) if str(param).strip()]
    if "." in symbol_name:
        class_name, method_name = symbol_name.split(".", 1)
        if keyword.iskeyword(class_name) or keyword.iskeyword(method_name):
            return
        _register_required_method(contract, class_name, method_name)
        if clean_params:
            contract["exact_params"][f"{class_name}.{method_name}"] = clean_params
        return
    contract["required_top_level"].add(symbol_name)
    if clean_params:
        contract["exact_params"][symbol_name] = clean_params


def _register_symbol_hint(contract: Dict[str, Any], raw_symbol: Any) -> None:
    symbol_text = str(raw_symbol or "").strip().strip("`'\"")
    if not symbol_text:
        return
    symbol_text = re.sub(
        r"^[A-Za-z][A-Za-z0-9_ /\-]*:\s*(?=[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?:\(|\b))",
        "",
        symbol_text,
    )
    symbol_text = re.sub(r"\s+", " ", symbol_text)
    signature_match = re.match(
        r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\(([^()]*)\)",
        symbol_text,
    )
    if signature_match:
        _register_signature(
            contract,
            signature_match.group(1),
            _extract_signature_param_names(signature_match.group(2)),
        )
        return
    method_match = re.match(r"^([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", symbol_text)
    if method_match:
        _register_required_method(contract, method_match.group(1), method_match.group(2))
        return
    identifier_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", symbol_text)
    if identifier_match:
        identifier = identifier_match.group(1)
        if not keyword.iskeyword(identifier):
            contract["required_top_level"].add(identifier)


def _apply_text_contract_hints(
    contract: Dict[str, Any],
    text: str,
    allow_free_functions: bool = False,
) -> None:
    if not text:
        return
    class_matches = set(re.findall(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    exported_symbols = _extract_exported_symbol_names(text)
    for class_name in class_matches:
        contract["required_top_level"].add(class_name)
    for symbol_name in exported_symbols:
        if symbol_name[:1].isupper():
            contract["required_top_level"].add(symbol_name)
    for class_name, method_name in _extract_class_method_refs(text):
        if class_name in class_matches or class_name in contract["required_top_level"]:
            _register_required_method(contract, class_name, method_name)
    for symbol_name, params in _extract_dotted_signatures(text).items():
        if "." in symbol_name:
            class_name = symbol_name.split(".", 1)[0]
            if class_name in class_matches or class_name in contract["required_top_level"]:
                _register_signature(contract, symbol_name, params)
        elif allow_free_functions:
            _register_signature(contract, symbol_name, params)
    if allow_free_functions:
        for func_name in _extract_expected_function_names(text):
            contract["required_top_level"].add(func_name)
    exported_classes = [name for name in exported_symbols if name[:1].isupper()]
    if len(exported_classes) == 1:
        match = re.search(r"(Exports?[^.]+)", text or "", flags=re.IGNORECASE)
        export_sentence = match.group(1) if match else ""
        lowered = export_sentence.lower()
        if " class with " in f" {lowered} " or " methods " in f" {lowered} ":
            owner = exported_classes[0]
            for symbol_name, params in _extract_dotted_signatures(export_sentence).items():
                if "." in symbol_name:
                    continue
                if keyword.iskeyword(symbol_name) or symbol_name in CONTRACT_TEXT_STOPWORDS:
                    continue
                _register_signature(contract, f"{owner}.{symbol_name}", params)


def _apply_symbol_index_hints(
    contract: Dict[str, Any],
    target_file: str,
    symbol_index: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(symbol_index, dict):
        return
    file_entry = ((symbol_index.get("files") or {}).get(target_file) or {})
    for symbol in file_entry.get("symbols", []) or []:
        if not isinstance(symbol, dict):
            continue
        name = str(symbol.get("name") or "").strip()
        kind = str(symbol.get("kind") or "").strip()
        if not name or name.startswith("__") and name.endswith("__"):
            continue
        if kind == "class":
            contract["required_top_level"].add(name)
        elif kind in {"function", "async_function"}:
            contract["required_top_level"].add(name)
            signature = str(symbol.get("signature") or "")
            match = re.match(r"^(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^()]*)\)", signature)
            if match:
                contract["exact_params"].setdefault(name, _extract_signature_param_names(match.group(1)))


def _build_task_entry_map(task_payload: Any) -> Dict[str, Dict[str, Any]]:
    entry_map: Dict[str, Dict[str, Any]] = {}
    if not isinstance(task_payload, dict):
        return entry_map
    candidate_lists = [
        task_payload.get("Logic_Task_List"),
        task_payload.get("logic_task_list"),
        task_payload.get("Task list"),
        task_payload.get("task_list"),
    ]
    for candidate in candidate_lists:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if not isinstance(item, dict):
                continue
            raw_path = (
                item.get("target_file")
                or item.get("file")
                or item.get("path")
                or item.get("relative_path")
            )
            clean_name = sanitize_todo_file_name(raw_path or "")
            if clean_name and clean_name not in entry_map:
                entry_map[clean_name] = item
    return entry_map


def _looks_like_time_series_case(target_file: str, file_analysis: str, config_yaml: str) -> bool:
    base = os.path.basename(target_file)
    if base not in STRICT_CONTRACT_FILES:
        return False
    config_text = str(config_yaml or "").lower()
    analysis_text = str(file_analysis or "").lower()
    config_signals = [
        "time series",
        "prediction horizon",
        "lookback",
        "num_variates",
    ]
    analysis_signals = [
        "forecast",
        "timeseriesdataset",
        "create_dataloaders",
        "transformerbaseline",
        "load_csv",
        "compute_scaler",
        "split_bounds",
    ]
    config_hits = sum(token in config_text for token in config_signals)
    analysis_hits = sum(token in analysis_text for token in analysis_signals)
    if not analysis_text.strip():
        return config_hits >= 2 and base in {
            "dataset_loader.py",
            "evaluation.py",
            "evaluator.py",
            "metrics.py",
            "model.py",
            "trainer.py",
        }
    return analysis_hits >= 1 and (analysis_hits + config_hits) >= 2


def _build_dynamic_stub_contract(
    target_file: str,
    file_analysis: str,
    task_entry: Optional[Dict[str, Any]] = None,
    feature_metadata: Optional[Dict[str, Any]] = None,
    closure: Optional[Dict[str, Any]] = None,
    context_bundle: Optional[Dict[str, Any]] = None,
    symbol_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    contract = _empty_stub_contract()
    base_name = os.path.basename(target_file)
    feature_metadata = feature_metadata if isinstance(feature_metadata, dict) else {}
    closure = closure if isinstance(closure, dict) else {}
    context_bundle = context_bundle if isinstance(context_bundle, dict) else {}
    task_entry = task_entry if isinstance(task_entry, dict) else {}

    explicit_symbols = []
    for source in (
        closure.get("target_symbols", []),
        context_bundle.get("target_symbols", []),
        task_entry.get("target_symbols", []),
        task_entry.get("replaced_symbols", []),
        task_entry.get("modified_symbols", []),
        task_entry.get("symbols", []),
    ):
        explicit_symbols.extend(source or [])
    for raw_symbol in explicit_symbols:
        _register_symbol_hint(contract, raw_symbol)

    text_sources: List[str] = [file_analysis]
    text_sources.extend(_flatten_string_fragments(task_entry))
    for per_file_map_name in (
        "callsite_updates_by_file",
        "public_interface_changes_by_file",
        "anti_simplification_by_file",
        "new_files_by_path",
    ):
        per_file_map = feature_metadata.get(per_file_map_name, {})
        if isinstance(per_file_map, dict):
            text_sources.extend(_flatten_string_fragments(per_file_map.get(target_file, [])))
    text_sources.extend(_flatten_string_fragments(closure))
    text_sources.extend(_flatten_string_fragments(context_bundle))

    allow_free_functions = base_name in {"utils.py", "metrics.py"}
    for text in text_sources:
        _apply_text_contract_hints(contract, text, allow_free_functions=allow_free_functions)

    _apply_symbol_index_hints(contract, target_file, symbol_index)
    return _normalize_stub_contract(contract)


def _build_stub_contract(
    target_file: str,
    file_analysis: str,
    config_yaml: str = "",
    task_entry: Optional[Dict[str, Any]] = None,
    feature_metadata: Optional[Dict[str, Any]] = None,
    closure: Optional[Dict[str, Any]] = None,
    context_bundle: Optional[Dict[str, Any]] = None,
    symbol_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dynamic_contract = _build_dynamic_stub_contract(
        target_file,
        file_analysis,
        task_entry=task_entry,
        feature_metadata=feature_metadata,
        closure=closure,
        context_bundle=context_bundle,
        symbol_index=symbol_index,
    )
    legacy_contract = _legacy_stub_contract(target_file, file_analysis)
    if _looks_like_time_series_case(target_file, file_analysis, config_yaml):
        return _merge_stub_contracts(legacy_contract, dynamic_contract)
    if _contract_has_requirements(dynamic_contract):
        return dynamic_contract
    return legacy_contract if os.path.basename(target_file) == "utils.py" else dynamic_contract


def _validate_stub_contract(
    target_file: str,
    contract: Dict[str, Any],
    stub_text: str,
) -> Tuple[bool, str]:
    contract = _normalize_stub_contract(contract)
    required_top_level: Set[str] = set(contract.get("required_top_level") or set())
    required_methods: Dict[str, Set[str]] = {
        key: set(value) for key, value in (contract.get("required_methods") or {}).items()
    }
    exact_params: Dict[str, List[str]] = dict(contract.get("exact_params") or {})
    if not required_top_level and not required_methods and not exact_params:
        return True, ""

    top_level_defs, methods_by_class = _extract_top_level_defs(stub_text)
    stub_signatures = extract_stub_signatures(stub_text)
    errors: List[str] = []

    missing_top_level = sorted(required_top_level - top_level_defs)
    if missing_top_level:
        errors.append(
            f"Stub is missing required top-level symbols for {os.path.basename(target_file)}: "
            f"{', '.join(missing_top_level)}."
        )

    for class_name, method_names in required_methods.items():
        available_methods = methods_by_class.get(class_name, set())
        missing_methods = sorted(method_names - available_methods)
        if missing_methods:
            errors.append(
                f"Stub is missing required methods on {class_name}: {', '.join(missing_methods)}."
            )

    for symbol_name, expected_params in exact_params.items():
        actual_params = stub_signatures.get(symbol_name)
        if actual_params is None:
            errors.append(f"Stub is missing required signature: {symbol_name}.")
            continue
        if actual_params != expected_params:
            errors.append(
                f"Stub signature mismatch for {symbol_name}: expected params {expected_params}, "
                f"got {actual_params}."
            )

    if contract.get("disallow_extra_public_top_level"):
        extras = sorted(
            name for name in top_level_defs - required_top_level
            if not name.startswith("_")
        )
        if extras:
            errors.append(
                f"Stub adds unexpected public top-level symbols for {os.path.basename(target_file)}: "
                f"{', '.join(extras)}."
            )

    if contract.get("disallow_extra_public_methods"):
        for class_name, available_methods in methods_by_class.items():
            allowed_methods = required_methods.get(class_name, set())
            extra_methods = sorted(
                name for name in available_methods - allowed_methods
                if not name.startswith("_")
            )
            if extra_methods:
                errors.append(
                    f"Stub adds unexpected public methods on {class_name}: {', '.join(extra_methods)}."
                )

    if errors:
        return False, " ".join(errors[:4])
    return True, ""


def _is_python_stub_target(file_name: str) -> bool:
    return file_name.endswith(".py")


def _should_use_review(file_name: str) -> bool:
    return os.path.basename(file_name) not in LIGHT_REVIEW_FILES


def _stub_storage_name(file_name: str) -> str:
    return quote(file_name, safe="._-") + ".py"


def _legacy_stub_storage_name(file_name: str) -> str:
    return file_name.replace("/", "_") + ".py"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt_path(prompt_set: Optional[str], name: str) -> str:
    return f"{prompt_set}/{name}" if prompt_set else name


def _build_upstream_stubs_text(
    target_file: str,
    rpg: PipelineRPG,
    stubs_dict: Dict[str, str],
) -> str:
    """Build text showing verified stubs of all dependencies."""
    deps = rpg.get_dependencies(target_file)
    if not deps:
        return "(no dependencies — this file has no upstream dependencies)"

    parts = []
    for dep in deps:
        if dep in stubs_dict:
            parts.append(
                f"### {dep} (VERIFIED stub — use these exact signatures)\n"
                f"```python\n{stubs_dict[dep]}\n```\n"
            )
        else:
            parts.append(f"### {dep} (stub not available — design independently)\n")

    return "\n".join(parts) if parts else "(no dependency stubs available)"


def _build_dependency_list_text(
    target_file: str,
    rpg: PipelineRPG,
) -> str:
    """Build a simple list of dependencies for the prompt."""
    deps = rpg.get_dependencies(target_file)
    if not deps:
        return "This file has no dependencies on other files."
    lines = [f"- {dep}" for dep in deps]
    dependents = rpg.get_dependents(target_file)
    if dependents:
        lines.append(f"\nFiles that will depend on this file: {', '.join(dependents)}")
        lines.append("Design the interface so these downstream files can use it easily.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load/build RPG
# ---------------------------------------------------------------------------

def _load_or_build_rpg(output_dir: str, gpt_version: str = None) -> PipelineRPG:
    rpg_path = os.path.join(output_dir, "rpg_graph.json")
    if os.path.exists(rpg_path):
        logger.info(f"  [RPG] Loading existing RPG from {rpg_path}")
        return PipelineRPG.load(rpg_path)
    else:
        logger.info(f"  [RPG] Building RPG from planning output")
        rpg = build_rpg_from_planning(output_dir, gpt_version=gpt_version)
        rpg.save(rpg_path)
        return rpg


# ---------------------------------------------------------------------------
# LLM Review Pass
# ---------------------------------------------------------------------------

def _review_stub(
    client,
    gpt_version: str,
    stub_code: str,
    target_file: str,
    dep_list: str,
    upstream_stubs: str,
    file_analysis: str,
    prompt_set: Optional[str],
) -> Tuple[bool, str, Optional[dict]]:
    """
    Run a second LLM pass to review the generated stub.

    Evaluates on 6 dimensions:
      1. Feature Alignment
      2. Structural Completeness
      3. Docstring Quality
      4. Interface Style
      5. Dependency Consistency
      6. ML Correctness

    Returns:
        (passed, feedback_text, review_json)
    """
    review_system = [{
        "role": "system",
        "content": render_prompt(
            _prompt_path(prompt_set, "interface_review_system.txt"),
        ),
    }]

    review_user = [{
        "role": "user",
        "content": render_prompt(
            _prompt_path(prompt_set, "interface_review_user.txt"),
            target_file=target_file,
            dependency_list=dep_list,
            upstream_stubs=upstream_stubs,
            file_analysis=file_analysis,
            stub_code=stub_code,
        ),
    }]

    review_msgs = review_system + review_user

    try:
        completion = chat_completion_with_retry(client, gpt_version, review_msgs)
        completion_json = json.loads(completion.model_dump_json())
        review_text = completion_json["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"  [REVIEW] LLM call failed: {exc} — skipping review")
        return True, "", None

    # Parse JSON from the review response
    try:
        # Try to extract JSON from the response
        review_text_clean = review_text.strip()
        # Handle markdown-wrapped JSON
        if review_text_clean.startswith("```json"):
            review_text_clean = review_text_clean[7:]
        elif review_text_clean.startswith("```"):
            review_text_clean = review_text_clean[3:]
        if review_text_clean.endswith("```"):
            review_text_clean = review_text_clean[:-3]
        review_text_clean = review_text_clean.strip()

        review_data = json.loads(review_text_clean)
    except json.JSONDecodeError:
        logger.warning(f"  [REVIEW] Could not parse review JSON — accepting stub")
        return True, "", None

    final_pass = review_data.get("final_pass", True)
    suggested_fixes = review_data.get("suggested_fixes", "")

    # Log review results
    review_details = review_data.get("review", {})
    failed_dims = [dim for dim, info in review_details.items()
                   if isinstance(info, dict) and not info.get("pass", True)]

    if final_pass:
        logger.info(f"    [REVIEW] ✓ Passed all dimensions")
    else:
        logger.info(f"    [REVIEW] ✗ Failed dimensions: {failed_dims}")
        logger.info(f"    [REVIEW] Suggested fixes: {suggested_fixes[:200]}")

    # Build feedback text for retry
    feedback = ""
    if not final_pass:
        feedback_parts = ["Review feedback (fix these issues):"]
        for dim, info in review_details.items():
            if isinstance(info, dict) and not info.get("pass", True):
                feedback_parts.append(f"  [{dim}]: {info.get('feedback', 'No detail')}")
        if suggested_fixes:
            feedback_parts.append(f"\nSuggested fixes: {suggested_fixes}")
        feedback = "\n".join(feedback_parts)

    return final_pass, feedback, review_data


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------

def run_interface_design(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    prompt_set: str = None,
) -> None:
    """
    Per-file interface stub generation stage.

    For each file in topological order:
      1. Build prompt with paper context + upstream verified stubs
      2. LLM generates stub (signatures + pass bodies)
      3. Validate with ast.parse() + structural checks
      4. Save to stubs/ directory

    This replaces the old api_predefine stage.
    """
    logger.info("=== Interface Design Stage (Per-File Stubs) ===")

    # ---------- Setup ----------
    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=16000)

    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    todo_file_lst = ctx.todo_file_lst
    task_list = ctx.task_list
    feature_metadata = ctx.feature_metadata if isinstance(ctx.feature_metadata, dict) else {}
    modification_closure_by_file = (
        ctx.modification_closure_by_file if isinstance(ctx.modification_closure_by_file, dict) else {}
    )
    context_bundle_by_file = (
        ctx.context_bundle_by_file if isinstance(ctx.context_bundle_by_file, dict) else {}
    )
    symbol_index = ctx.symbol_index if isinstance(ctx.symbol_index, dict) else {}

    # ---------- Load RPG ----------
    rpg = _load_or_build_rpg(output_dir, gpt_version)
    rpg_sorted = rpg.topological_sort()

    # Only process files in todo list
    original_set = set(todo_file_lst)
    file_order = [f for f in rpg_sorted if f in original_set]
    # Add any files RPG missed
    for f in todo_file_lst:
        if f not in set(file_order):
            file_order.append(f)

    logger.info(f"  File order for stub generation ({len(file_order)} files):")
    for i, f in enumerate(file_order):
        deps = rpg.get_dependencies(f)
        dep_str = f" <- [{', '.join(deps)}]" if deps else ""
        logger.info(f"    {i+1}. {f}{dep_str}")

    stub_target_files = [f for f in file_order if _is_python_stub_target(f)]
    skipped_stub_targets = [f for f in file_order if f not in set(stub_target_files)]
    if skipped_stub_targets:
        logger.info(
            "  Skipping non-Python files for stub design: "
            + ", ".join(skipped_stub_targets)
        )

    # ---------- Load existing analyses ----------
    analysis_dict: Dict[str, str] = dict(ctx.logic_analysis_dict)
    for fname in file_order:
        if fname == "config.yaml":
            continue
        save_name = sanitize_todo_file_name(fname).replace("/", "_")
        analysis_path = os.path.join(output_dir, f"{save_name}_simple_analysis.txt")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_dict[sanitize_todo_file_name(fname)] = f.read()

    task_entry_map = _build_task_entry_map(task_list)
    contract_by_file: Dict[str, Dict[str, Any]] = {}
    for fname in stub_target_files:
        clean_name = sanitize_todo_file_name(fname)
        if not clean_name:
            continue
        contract_by_file[clean_name] = _build_stub_contract(
            clean_name,
            analysis_dict.get(clean_name, ""),
            config_yaml=config_yaml,
            task_entry=task_entry_map.get(clean_name),
            feature_metadata=feature_metadata,
            closure=modification_closure_by_file.get(clean_name),
            context_bundle=context_bundle_by_file.get(clean_name),
            symbol_index=symbol_index,
        )

    # ---------- Stubs directory ----------
    stubs_dir = os.path.join(output_dir, "stubs")
    os.makedirs(stubs_dir, exist_ok=True)

    artifact_dir = os.path.join(output_dir, "interface_design_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # ---------- Track verified stubs ----------
    stubs_dict: Dict[str, str] = {}  # file_name -> verified stub code
    stub_status: Dict[str, Dict[str, Any]] = {}

    for fname in skipped_stub_targets:
        clean_name = sanitize_todo_file_name(fname)
        if not clean_name:
            continue
        stub_status[clean_name] = {
            "status": STUB_STATUS_SKIPPED,
            "reason": "non_python_target",
            "storage_name": "",
            "signatures": {},
        }

    # Load any already-generated stubs (for resume support)
    for fname in stub_target_files:
        clean_name = sanitize_todo_file_name(fname)
        storage_name = _stub_storage_name(clean_name)
        candidates = [
            os.path.join(stubs_dir, storage_name),
            os.path.join(stubs_dir, _legacy_stub_storage_name(clean_name)),
        ]
        loaded = False
        for stub_path in candidates:
            if not os.path.exists(stub_path):
                continue
            with open(stub_path, "r", encoding="utf-8") as f:
                existing_stub = f.read()
            if _is_fallback_stub(existing_stub):
                stub_status[clean_name] = {
                    "status": STUB_STATUS_FAILED,
                    "reason": "fallback_stub",
                    "storage_name": os.path.basename(stub_path),
                    "signatures": {},
                }
                logger.info(f"  [RETRY] Ignoring fallback stub for {clean_name}")
                loaded = True
                break
            ok, err_msg = validate_stub(existing_stub)
            if not ok:
                stub_status[clean_name] = {
                    "status": STUB_STATUS_FAILED,
                    "reason": f"invalid_existing_stub: {err_msg}",
                    "storage_name": os.path.basename(stub_path),
                    "signatures": {},
                }
                logger.info(f"  [RETRY] Ignoring invalid existing stub for {clean_name}: {err_msg}")
                loaded = True
                break
            contract_ok, contract_err = _validate_stub_contract(
                clean_name,
                contract_by_file.get(clean_name, _empty_stub_contract()),
                existing_stub,
            )
            if not contract_ok:
                stub_status[clean_name] = {
                    "status": STUB_STATUS_FAILED,
                    "reason": f"existing_stub_contract_error: {contract_err}",
                    "storage_name": os.path.basename(stub_path),
                    "signatures": {},
                }
                logger.info(f"  [RETRY] Ignoring contract-mismatched stub for {clean_name}: {contract_err}")
                loaded = True
                break
            stubs_dict[clean_name] = existing_stub
            stub_status[clean_name] = {
                "status": STUB_STATUS_VERIFIED,
                "reason": "loaded_existing_stub",
                "storage_name": storage_name,
                "signatures": extract_stub_signatures(existing_stub),
            }
            logger.info(f"  [SKIP] Loaded existing verified stub: {clean_name}")
            loaded = True
            break
        if not loaded:
            stub_status.setdefault(clean_name, {
                "status": STUB_STATUS_FAILED,
                "reason": "missing_stub",
                "storage_name": storage_name,
                "signatures": {},
            })

    # ---------- System prompt ----------
    system_msg = [{
        "role": "system",
        "content": render_prompt(
            _prompt_path(prompt_set, "interface_design_system.txt"),
            paper_format=paper_format,
        ),
    }]

    # ---------- Cost tracking ----------
    total_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")

    # ---------- Main loop ----------
    for idx, fname in enumerate(file_order):
        clean_name = sanitize_todo_file_name(fname)
        if not clean_name or clean_name == "config.yaml":
            continue
        if not _is_python_stub_target(clean_name):
            continue

        # Skip if already done
        if clean_name in stubs_dict:
            continue

        current_stage = f"[INTERFACE_DESIGN] {clean_name}"
        logger.info(f"  [{idx+1}/{len(file_order)}] Designing stub for: {clean_name}")

        # Build prompt
        upstream_stubs = _build_upstream_stubs_text(clean_name, rpg, stubs_dict)
        dep_list = _build_dependency_list_text(clean_name, rpg)
        file_analysis = analysis_dict.get(clean_name, "(no analysis available)")

        user_msg = [{
            "role": "user",
            "content": render_prompt(
                _prompt_path(prompt_set, "interface_design_user.txt"),
                paper_content=paper_content_prompt,
                config_yaml=config_yaml,
                target_file=clean_name,
                dependency_list=dep_list,
                upstream_stubs=upstream_stubs,
                file_analysis=file_analysis,
            ),
        }]

        trajectories = copy.deepcopy(system_msg)
        trajectories.extend(user_msg)

        # Generate with retries
        final_stub = ""
        last_error = ""
        completion = None

        for attempt in range(3):
            try:
                completion = chat_completion_with_retry(client, gpt_version, trajectories)
            except Exception as exc:
                logger.error(f"  API call failed: {exc}")
                raise

            completion_json = json.loads(completion.model_dump_json())
            model_text = completion_json["choices"][0]["message"]["content"]

            # Extract code
            stub_text = extract_code_from_content(model_text)
            if not stub_text:
                stub_text = model_text
            stub_text = _normalize_stub_text(stub_text)

            # Step 1: AST + structural validation
            ok, err_msg = validate_stub(stub_text)
            if not ok:
                last_error = err_msg
                logger.warning(
                    f"  [RETRY {attempt+1}/3] Stub validation failed for {clean_name}: {err_msg}"
                )
                trajectories.append({"role": "assistant", "content": model_text})
                trajectories.append({
                    "role": "user",
                    "content": (
                        f"Your stub is invalid.\n"
                        f"Validation error: {err_msg}\n\n"
                        f"Return ONLY one Python code block containing a valid stub.\n"
                        f"ALL function/method bodies must be exactly `pass`.\n"
                        f"Fix the issues and try again."
                    ),
                })
                continue

            # Step 1.5: deterministic contract validation
            ok, err_msg = _validate_stub_contract(
                clean_name,
                contract_by_file.get(clean_name, _empty_stub_contract()),
                stub_text,
            )
            if not ok:
                last_error = err_msg
                logger.warning(
                    f"  [RETRY {attempt+1}/3] Stub contract validation failed for {clean_name}: {err_msg}"
                )
                trajectories.append({"role": "assistant", "content": model_text})
                trajectories.append({
                    "role": "user",
                    "content": (
                        f"Your stub is structurally valid but does not match the required contract.\n"
                        f"Contract error: {err_msg}\n\n"
                        "Keep ONLY the required public API for this file.\n"
                        "Do not invent extra public symbols, helper factories, or alternate signatures.\n"
                        "Return ONLY one corrected Python code block."
                    ),
                })
                continue

            # Step 2: LLM Review Pass
            review_passed = True
            review_feedback = ""
            review_data = None
            if _should_use_review(clean_name):
                review_passed, review_feedback, review_data = _review_stub(
                    client, gpt_version, stub_text, clean_name,
                    dep_list, upstream_stubs, file_analysis, prompt_set,
                )
            else:
                logger.info(f"    [REVIEW] Skipped lightweight review for {clean_name}")

            # Save review artifact
            if review_data:
                review_path = os.path.join(
                    artifact_dir, f"{clean_name.replace('/', '_')}_review.json"
                )
                with open(review_path, "w", encoding="utf-8") as rf:
                    json.dump(review_data, rf, indent=2)

            # Log review cost
            total_cost = print_log_cost(
                completion_json, gpt_version,
                f"[INTERFACE_REVIEW] {clean_name}",
                output_dir, total_cost,
            )

            if review_passed:
                final_stub = stub_text
                last_error = ""
                break

            # Review failed — feed feedback back for retry
            last_error = f"Review failed: {review_feedback[:200]}"
            logger.warning(
                f"  [RETRY {attempt+1}/3] Review rejected stub for {clean_name}"
            )
            trajectories.append({"role": "assistant", "content": model_text})
            trajectories.append({
                "role": "user",
                "content": (
                    f"Your stub passed syntax validation but failed quality review.\n\n"
                    f"{review_feedback}\n\n"
                    f"Return an improved Python code block with the fixes applied.\n"
                    f"ALL function/method bodies must be exactly `pass`."
                ),
            })

        storage_name = _stub_storage_name(clean_name)
        if not final_stub:
            logger.error(
                f"  [FAIL] Could not generate valid stub for {clean_name} "
                f"after 3 attempts: {last_error}"
            )
            stub_status[clean_name] = {
                "status": STUB_STATUS_FAILED,
                "reason": last_error or "generation_failed",
                "storage_name": storage_name,
                "signatures": {},
            }
        else:
            stubs_dict[clean_name] = final_stub
            stub_status[clean_name] = {
                "status": STUB_STATUS_VERIFIED,
                "reason": "generated",
                "storage_name": storage_name,
                "signatures": extract_stub_signatures(final_stub),
            }

            stub_path = os.path.join(stubs_dir, storage_name)
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "w", encoding="utf-8") as f:
                f.write(final_stub)
                f.write("\n")

        # Save artifact
        stub_save_name = clean_name.replace("/", "_")
        artifact_path = os.path.join(artifact_dir, f"{stub_save_name}_stub.txt")
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(completion_json["choices"][0]["message"]["content"])

        # Log cost
        print_response(completion_json)
        total_cost = print_log_cost(
            completion_json, gpt_version, current_stage,
            output_dir, total_cost,
        )

        # Log signatures
        if final_stub:
            sigs = extract_stub_signatures(final_stub)
            logger.info(f"    Signatures: {list(sigs.keys())}")

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_cost)

    # ---------- Save combined stubs for reference ----------
    combined_path = os.path.join(output_dir, "interface_stubs_combined.py")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("# Combined Interface Stubs (auto-generated)\n")
        f.write("# Each section below is a verified stub for one file.\n\n")
        for fname in file_order:
            clean = sanitize_todo_file_name(fname)
            meta = stub_status.get(clean, {})
            if meta.get("status") == STUB_STATUS_VERIFIED and clean in stubs_dict:
                f.write(f"\n# {'='*60}\n")
                f.write(f"# {clean}\n")
                f.write(f"# {'='*60}\n\n")
                f.write(stubs_dict[clean])
                f.write("\n\n")
        missing_or_skipped = [
            (fname, meta) for fname, meta in stub_status.items()
            if meta.get("status") != STUB_STATUS_VERIFIED
        ]
        if missing_or_skipped:
            f.write("\n# Missing or skipped stub targets\n")
            for fname, meta in sorted(missing_or_skipped):
                reason = meta.get("reason", "")
                status = meta.get("status", "unknown")
                f.write(f"# - {fname}: {status}")
                if reason:
                    f.write(f" ({reason})")
                f.write("\n")

    # Save stubs index
    index_path = os.path.join(stubs_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": STUB_INDEX_VERSION,
            "file_order": file_order,
            "stub_targets": stub_target_files,
            "stubs": {k: extract_stub_signatures(v) for k, v in stubs_dict.items()},
            "statuses": stub_status,
            "task_list_keys": sorted(task_list.keys()) if isinstance(task_list, dict) else [],
        }, f, indent=2)

    logger.info(f"  [DONE] {len(stubs_dict)} stubs saved to {stubs_dir}")
    logger.info(f"  [DONE] Combined stubs: {combined_path}")
    logger.info("=== Interface Design Stage Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from core.logger import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="RPG Interface Design (Per-File Stubs)")
    parser.add_argument("--paper_name", type=str, default="")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--paper_format", type=str, default="JSON")
    parser.add_argument("--pdf_json_path", type=str, default=None)
    parser.add_argument("--pdf_latex_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    cleaned = args.pdf_json_path
    if cleaned and not cleaned.endswith("_cleaned.json"):
        cleaned = cleaned.replace(".json", "_cleaned.json")

    run_interface_design(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format=args.paper_format,
        pdf_json_path=cleaned,
        pdf_latex_path=args.pdf_latex_path,
        prompt_set="baseline",
    )
