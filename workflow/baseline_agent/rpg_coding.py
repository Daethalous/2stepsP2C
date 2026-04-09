"""
RPG-Enhanced Coding — Phase 2 of the RPG integration.

Replaces the flat `build_code_interface_summary()` approach with
graph-aware context building that uses the RPG dependency graph to
provide focused, relevant code context for each file being generated.

Key improvements over the original coding.py:
  1. Uses RPG-sorted file order (from build_rpg.py)
  2. Shows FULL CODE of direct dependencies (not just signatures of everything)
  3. Shows SIGNATURES ONLY of siblings and transitive deps
  4. Adds cross-file signature validation after each generation
  5. Injects dependency metadata into the prompt

Usage:
    This module is designed to be called from the pipeline, replacing
    the original coding.py's context-building and file ordering.

    It can also be tested standalone:
        python -m workflow.baseline_agent.rpg_coding --output_dir <dir> --dry_run
"""

import ast
import copy
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.llm_engine import chat_completion_with_retry, create_client
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.utils import (
    build_code_interface_summary,
    contains_forbidden_placeholders,
    extract_code_from_content,
    extract_interface_signatures,
    format_paper_content_for_prompt,
    load_accumulated_cost,
    print_log_cost,
    print_response,
    save_accumulated_cost,
)

from workflow.baseline_agent.build_rpg import PipelineRPG, build_rpg_from_planning
from workflow.baseline_agent.rpg_adapter import get_coding_context, get_stub_context, load_stubs_dict
from workflow.baseline_agent.rpg_interface_design import extract_stub_signatures

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cross-file signature validation
# ---------------------------------------------------------------------------

def _parse_function_signatures(code: str) -> Dict[str, List[str]]:
    """
    Parse function/method signatures from Python code using AST.

    Returns:
        Dict mapping "function_name" -> [param_names]
        For methods, key is "ClassName.method_name"
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
                if arg.arg != "self" and arg.arg != "cls":
                    params.append(arg.arg)
            # Add *args, **kwargs
            if node.args.vararg:
                params.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                params.append(f"**{node.args.kwarg.arg}")

            # Check if method (nested inside a class)
            func_name = node.name
            signatures[func_name] = params

        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = []
                    for arg in item.args.args:
                        if arg.arg != "self" and arg.arg != "cls":
                            params.append(arg.arg)
                    if item.args.vararg:
                        params.append(f"*{item.args.vararg.arg}")
                    if item.args.kwarg:
                        params.append(f"**{item.args.kwarg.arg}")

                    key = f"{node.name}.{item.name}"
                    signatures[key] = params

    return signatures


def _extract_function_calls(code: str) -> List[Tuple[str, int]]:
    """
    Extract function/method calls and their argument counts from code.

    Returns:
        List of (function_name, num_args) tuples
    """
    calls = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return calls

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Get function name
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name:
                num_args = len(node.args) + len(node.keywords)
                calls.append((func_name, num_args))

    return calls


def validate_cross_file_signatures(
    new_code: str,
    new_file: str,
    done_file_dict: Dict[str, str],
    rpg: PipelineRPG,
) -> List[str]:
    """
    Validate that function calls in new_code match signatures defined
    in dependency files.

    Returns:
        List of warning messages about potential mismatches.
    """
    warnings = []

    # Parse calls in new code
    calls_in_new = _extract_function_calls(new_code)
    if not calls_in_new:
        return warnings

    # Collect signatures from dependency files
    dep_files = rpg.get_dependencies(new_file)
    all_dep_signatures = {}
    for dep in dep_files:
        dep_code = done_file_dict.get(dep, "")
        if dep_code:
            sigs = _parse_function_signatures(dep_code)
            for func_name, params in sigs.items():
                all_dep_signatures[func_name] = (dep, params)

    # Cross-check: for each call in new code, see if we know its signature
    for func_name, num_call_args in calls_in_new:
        if func_name in all_dep_signatures:
            dep_file, expected_params = all_dep_signatures[func_name]
            # Count required params (those without defaults)
            # This is approximate — we don't track defaults in our simple parser
            # but at minimum, num_call_args shouldn't exceed total params
            # (unless there are *args/**kwargs)
            has_varargs = any(p.startswith("*") for p in expected_params)
            required_count = len([p for p in expected_params if not p.startswith("*")])

            if not has_varargs and num_call_args > len(expected_params):
                warnings.append(
                    f"MISMATCH: {new_file} calls {func_name}() with {num_call_args} args, "
                    f"but {dep_file} defines it with {len(expected_params)} params: {expected_params}"
                )
            # Check too few args (rough heuristic)
            elif num_call_args < required_count and not has_varargs:
                # Only warn if significantly fewer (could have defaults)
                if num_call_args < required_count - 2:
                    warnings.append(
                        f"POSSIBLE MISMATCH: {new_file} calls {func_name}() with {num_call_args} args, "
                        f"but {dep_file} defines {required_count} params (some may have defaults): {expected_params}"
                    )

    return warnings


# ---------------------------------------------------------------------------
# RPG-enhanced context building
# ---------------------------------------------------------------------------

def build_rpg_coding_context(
    rpg: PipelineRPG,
    target_file: str,
    done_file_dict: Dict[str, str],
    done_file_lst: List[str],
    max_total_chars: int = 14000,
) -> str:
    """
    Build coding context using RPG dependency awareness.

    This replaces build_code_interface_summary() with a graph-aware version:
    - Direct dependencies get FULL CODE (so the LLM sees exact signatures)
    - Sibling files get SIGNATURES ONLY
    - Unrelated files are excluded entirely

    Falls back to the original flat approach if the RPG has no dependency
    information for the target file.
    """
    # Check if RPG has useful dependency info
    deps = rpg.get_dependencies(target_file)
    siblings = rpg.get_same_subtree_files(target_file)

    if not deps and not siblings:
        # No graph info available — fall back to original behavior
        logger.info(f"  [RPG] No dependency info for {target_file}, using flat context")
        return build_code_interface_summary(done_file_dict, done_file_lst, max_total_chars)

    # Build RPG-aware context
    context = get_coding_context(rpg, target_file, done_file_dict, max_total_chars)

    # Add dependency metadata header
    dep_info = (
        f"\n[DEPENDENCY INFO for {target_file}]\n"
        f"Direct dependencies: {deps}\n"
        f"Sibling files (same directory): {[s for s in siblings if s in done_file_dict]}\n"
        f"IMPORTANT: Match function signatures EXACTLY as defined in dependency files above.\n"
    )

    return dep_info + context


def _validate_against_stub(
    generated_code: str,
    target_file: str,
    own_stub: str,
) -> List[str]:
    """
    Compare generated code against its interface stub.

    Checks that all functions/classes defined in the stub exist in the
    generated code with matching parameter names.

    Returns:
        List of mismatch warnings (empty if all match).
    """
    if not own_stub:
        return []

    stub_sigs = extract_stub_signatures(own_stub)
    code_sigs = extract_stub_signatures(generated_code)

    warnings = []
    for func_name, stub_params in stub_sigs.items():
        if func_name not in code_sigs:
            warnings.append(
                f"MISSING: Stub defines '{func_name}' but it is not in the generated code."
            )
            continue

        code_params = code_sigs[func_name]
        # Compare parameter names (ignoring *args/**kwargs)
        stub_required = [p for p in stub_params if not p.startswith("*")]
        code_required = [p for p in code_params if not p.startswith("*")]

        if stub_required != code_required:
            warnings.append(
                f"PARAM MISMATCH: '{func_name}' stub has params {stub_required} "
                f"but generated code has {code_required}. "
                f"You MUST use the exact parameter names from the stub."
            )

    return warnings


def format_signature_warnings(warnings: List[str]) -> str:
    """Format signature mismatch warnings for injection into retry prompt."""
    if not warnings:
        return ""
    header = "\n[SIGNATURE MISMATCH DETECTED]\n"
    body = "\n".join(f"  - {w}" for w in warnings)
    footer = "\nPlease fix these mismatches by checking the dependency file signatures above.\n"
    return header + body + footer


# ---------------------------------------------------------------------------
# Load/save RPG for pipeline integration
# ---------------------------------------------------------------------------

def _load_or_build_rpg(output_dir: str) -> PipelineRPG:
    """Load existing RPG from disk, or build it from planning output."""
    rpg_path = os.path.join(output_dir, "rpg_graph.json")
    if os.path.exists(rpg_path):
        logger.info(f"  [RPG] Loading existing RPG from {rpg_path}")
        return PipelineRPG.load(rpg_path)
    else:
        logger.info(f"  [RPG] Building RPG from planning output")
        rpg = build_rpg_from_planning(output_dir)
        rpg.save(rpg_path)
        return rpg


def _load_global_api_contract_stub(output_dir: str) -> str:
    """Load the global API contract stub if it exists (legacy fallback)."""
    # First try new combined stubs
    combined_path = os.path.join(output_dir, "interface_stubs_combined.py")
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as fh:
            return fh.read()
    # Fall back to old .pyi format
    contract_path = os.path.join(output_dir, "api_predefine_contract.pyi")
    if os.path.exists(contract_path):
        with open(contract_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "(no global API contract available)"


def _is_entry_point_file(file_name: str) -> bool:
    """Check if a file is an entry point that needs integration hints."""
    bn = os.path.basename(file_name).lower()
    return bn in (
        "main.py", "train.py", "run.py", "__init__.py",
        "experiment.py", "evaluate.py",
    )


def _extract_new_definitions(code: str) -> List[str]:
    """Extract class/function definitions from generated code."""
    defs = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") or stripped.startswith("def "):
            defs.append(stripped.split("(")[0].split(":")[0])
    return defs


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


# ---------------------------------------------------------------------------
# Enhanced coding stage
# ---------------------------------------------------------------------------

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
) -> None:
    """
    RPG-enhanced coding stage.

    This is a drop-in replacement for workflow.coding.run_coding() that:
    1. Uses RPG-sorted file order
    2. Builds graph-aware context (full code for deps, signatures for rest)
    3. Validates cross-file signatures after each generation
    4. Injects signature warnings into retry prompts
    """
    logger.info("=== RPG-Enhanced Coding Stage ===")

    # ---------- Setup ----------
    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=22000)

    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    context_lst = ctx.context_lst
    overview_prompt = format_paper_content_for_prompt(context_lst[0], max_chars=12000) if len(context_lst) > 0 else ""
    design_prompt = format_paper_content_for_prompt(context_lst[1], max_chars=16000) if len(context_lst) > 1 else ""
    task_prompt = format_paper_content_for_prompt(context_lst[2], max_chars=16000) if len(context_lst) > 2 else ""
    global_api_contract_stub = _load_global_api_contract_stub(output_dir)

    # ---------- Load Interface Stubs ----------
    stubs_dict = load_stubs_dict(output_dir)
    has_stubs = len(stubs_dict) > 0
    if has_stubs:
        logger.info(f"  [RPG] Loaded {len(stubs_dict) // 2} interface stubs for enforcement")
    else:
        logger.info("  [RPG] No interface stubs found — skipping stub enforcement")

    # ---------- Load/Build RPG ----------
    rpg = _load_or_build_rpg(output_dir)

    # Use RPG-sorted order instead of original todo_file_lst
    original_order = ctx.todo_file_lst
    rpg_sorted_order = rpg.topological_sort()

    # Only include files that are in the original todo list
    todo_file_lst = [f for f in rpg_sorted_order if f in set(original_order)]

    # Add any files from original that RPG might have missed
    rpg_set = set(todo_file_lst)
    for f in original_order:
        if f not in rpg_set:
            todo_file_lst.append(f)

    logger.info(f"  File order: RPG-sorted ({len(todo_file_lst)} files)")
    for i, f in enumerate(todo_file_lst):
        deps = rpg.get_dependencies(f)
        dep_str = f" <- [{', '.join(deps)}]" if deps else ""
        logger.info(f"    {i+1}. {f}{dep_str}")

    done_file_lst = ["config.yaml"]
    done_file_dict = {}
    new_definitions = []

    # ---------- Load existing analysis ----------
    detailed_logic_analysis_dict = {}
    for todo_file_name in todo_file_lst:
        if todo_file_name == "config.yaml":
            continue
        save_name = sanitize_todo_file_name(todo_file_name).replace("/", "_")
        analysis_path = os.path.join(output_dir, f"{save_name}_simple_analysis.txt")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                detailed_logic_analysis_dict[sanitize_todo_file_name(todo_file_name)] = f.read()

    # ---------- Build prompts ----------
    code_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "coding_system.txt"),
            paper_format=paper_format)}
    ]

    def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst):
        # KEY CHANGE: Use RPG-aware context instead of flat interface dump
        code_files = build_rpg_coding_context(
            rpg, todo_file_name, done_file_dict, done_file_lst,
            max_total_chars=14000,
        )

        extra_kwargs = {}
        if _is_entry_point_file(todo_file_name) and new_definitions:
            extra_kwargs["integration_hint"] = (
                "You have created/updated these definitions in earlier files:\n"
                + "\n".join(f"- {x}" for x in new_definitions[:50])
                + "\nEnsure this entry/factory/registry file integrates them where required."
            )

        # Build stub context for this file
        stub_contract = global_api_contract_stub  # fallback
        if has_stubs:
            own_stub, dep_stubs = get_stub_context(stubs_dict, todo_file_name, rpg)
            if own_stub:
                stub_contract = (
                    f"## Interface Contract for {todo_file_name}\n"
                    f"You MUST implement exactly these signatures. "
                    f"Do NOT rename parameters or change return types.\n"
                    f"```python\n{own_stub}\n```\n"
                )
                if dep_stubs:
                    stub_contract += (
                        f"\n## Dependency Interfaces (VERIFIED — use these, do NOT reimplement)\n"
                        f"{dep_stubs}"
                    )

        write_msg = [
            {"role": "user", "content": render_prompt(
                _prompt_path(prompt_set, "coding_user.txt"),
                paper_content=paper_content_prompt,
                overview=overview_prompt,
                design=design_prompt,
                task=task_prompt,
                config_yaml=config_yaml,
                code_files=code_files,
                todo_file_name=todo_file_name,
                done_file_lst=done_file_lst,
                detailed_logic_analysis=detailed_logic_analysis,
                global_api_contract_stub=stub_contract,
                **extra_kwargs)}
        ]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    # ---------- Main coding loop ----------
    artifact_output_dir = f"{output_dir}/coding_artifacts"
    os.makedirs(artifact_output_dir, exist_ok=True)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    signature_report = []  # Collect all sig warnings for final report

    for todo_idx, todo_file_name in enumerate(todo_file_lst):
        responses = []
        trajectories = copy.deepcopy(code_msg)

        current_stage = f"[RPG_CODING] {todo_file_name}"
        logger.info(current_stage)

        if todo_file_name == "config.yaml":
            continue

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        if len(clean_todo_file_name.strip()) == 0:
            logger.warning(f"  [SKIP] empty/invalid todo file name: {todo_file_name}")
            continue

        # Skip if already done
        save_todo_file_name_skip = clean_todo_file_name.replace("/", "_")
        skip_path = f"{artifact_output_dir}/{save_todo_file_name_skip}_coding.txt"
        if os.path.exists(skip_path):
            repo_file_path = os.path.join(output_repo_dir, clean_todo_file_name)
            if os.path.exists(repo_file_path):
                with open(repo_file_path, "r", encoding="utf-8") as f:
                    done_file_dict[clean_todo_file_name] = f.read()
            done_file_lst.append(clean_todo_file_name)
            logger.info(f"  [SKIP] artifact already exists: {skip_path}")
            continue

        # Log dependency info
        deps = rpg.get_dependencies(clean_todo_file_name)
        if deps:
            available = [d for d in deps if d in done_file_dict]
            missing = [d for d in deps if d not in done_file_dict]
            logger.info(f"  [RPG] Dependencies: available={available}, missing={missing}")

        # Build prompt
        instruction_msg = get_write_msg(
            clean_todo_file_name,
            detailed_logic_analysis_dict.get(clean_todo_file_name, ""),
            done_file_lst,
        )
        trajectories.extend(instruction_msg)

        # Generate code with retry
        completion = None
        for attempt in range(3):
            try:
                completion = api_call(trajectories)
            except Exception as api_exc:
                logger.error(f"  API call failed: {api_exc}")
                raise

            completion_json_try = json.loads(completion.model_dump_json())
            content_try = completion_json_try["choices"][0]["message"]["content"]
            code_try = extract_code_from_content(content_try) or content_try

            # Check for placeholders
            if contains_forbidden_placeholders(code_try):
                trajectories.append({"role": "assistant", "content": content_try})
                trajectories.append({
                    "role": "user",
                    "content": (
                        "Output rejected: code still contains NotImplementedError/TODO/placeholder. "
                        "Return ONLY the complete corrected file code, no explanations."
                    ),
                })
                logger.warning(f"  [RETRY {attempt+1}/3] Placeholder detected in {clean_todo_file_name}")
                continue

            # Cross-file signature validation
            sig_warnings = validate_cross_file_signatures(
                code_try, clean_todo_file_name, done_file_dict, rpg
            )

            # Stub contract validation (NEW)
            if has_stubs:
                own_stub, _ = get_stub_context(stubs_dict, clean_todo_file_name, rpg)
                stub_warnings = _validate_against_stub(code_try, clean_todo_file_name, own_stub)
                sig_warnings.extend(stub_warnings)

            if sig_warnings and attempt < 2:
                # Inject warnings into retry prompt
                warning_text = format_signature_warnings(sig_warnings)
                trajectories.append({"role": "assistant", "content": content_try})
                trajectories.append({
                    "role": "user",
                    "content": (
                        f"Potential issues detected in your code:\n{warning_text}\n"
                        "Please fix any function call mismatches to match the exact signatures "
                        "defined in the dependency files and your interface contract. "
                        "Return the complete corrected code."
                    ),
                })
                logger.warning(f"  [RETRY {attempt+1}/3] Validation issues in {clean_todo_file_name}: {sig_warnings}")
                for w in sig_warnings:
                    signature_report.append(f"{clean_todo_file_name}: {w}")
                continue

            # If we get here, code is clean
            if sig_warnings:
                # Still has warnings on last attempt — log but accept
                for w in sig_warnings:
                    logger.warning(f"  [SIG WARNING] {w}")
                    signature_report.append(f"{clean_todo_file_name}: {w} (accepted)")
            break

        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)

        message = completion.choices[0].message
        trajectories.append({"role": message.role, "content": message.content})

        done_file_lst.append(clean_todo_file_name)

        os.makedirs(output_repo_dir, exist_ok=True)
        save_todo_file_name = clean_todo_file_name.replace("/", "_")

        print_response(completion_json)
        total_accumulated_cost = print_log_cost(
            completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost
        )

        with open(f"{artifact_output_dir}/{save_todo_file_name}_coding.txt", "w", encoding="utf-8") as f:
            f.write(completion_json["choices"][0]["message"]["content"])

        code = extract_code_from_content(message.content)
        if len(code) == 0:
            code = message.content

        done_file_dict[clean_todo_file_name] = code
        new_definitions.extend(_extract_new_definitions(code))

        file_path = os.path.join(output_repo_dir, clean_todo_file_name)
        if os.path.isdir(file_path):
            logger.warning(f"  [SKIP] target path is a directory: {file_path}")
            continue
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

    # Save signature validation report
    if signature_report:
        report_path = os.path.join(artifact_output_dir, "rpg_signature_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("RPG Cross-File Signature Validation Report\n")
            f.write("=" * 50 + "\n\n")
            for entry in signature_report:
                f.write(f"  {entry}\n")
        logger.info(f"  [RPG] Signature report saved to {report_path}")
        logger.warning(f"  [RPG] {len(signature_report)} signature warnings detected")
    else:
        logger.info("  [RPG] No signature mismatches detected!")

    logger.info("=== RPG-Enhanced Coding Complete ===")
