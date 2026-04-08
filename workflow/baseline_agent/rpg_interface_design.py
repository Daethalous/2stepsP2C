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
import os
import re
from typing import Dict, List, Optional, Tuple

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

    # ---------- Load existing analyses ----------
    analysis_dict: Dict[str, str] = {}
    for fname in file_order:
        if fname == "config.yaml":
            continue
        save_name = sanitize_todo_file_name(fname).replace("/", "_")
        analysis_path = os.path.join(output_dir, f"{save_name}_simple_analysis.txt")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_dict[sanitize_todo_file_name(fname)] = f.read()

    # ---------- Stubs directory ----------
    stubs_dir = os.path.join(output_dir, "stubs")
    os.makedirs(stubs_dir, exist_ok=True)

    artifact_dir = os.path.join(output_dir, "interface_design_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # ---------- Track verified stubs ----------
    stubs_dict: Dict[str, str] = {}  # file_name -> verified stub code

    # Load any already-generated stubs (for resume support)
    for fname in file_order:
        clean_name = sanitize_todo_file_name(fname)
        stub_path = os.path.join(stubs_dir, clean_name.replace("/", "_") + ".py")
        if os.path.exists(stub_path):
            with open(stub_path, "r", encoding="utf-8") as f:
                stubs_dict[clean_name] = f.read()
            logger.info(f"  [SKIP] Loaded existing stub: {clean_name}")

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

            # Step 2: LLM Review Pass
            review_passed, review_feedback, review_data = _review_stub(
                client, gpt_version, stub_text, clean_name,
                dep_list, upstream_stubs, file_analysis, prompt_set,
            )

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

        if not final_stub:
            logger.error(
                f"  [FAIL] Could not generate valid stub for {clean_name} "
                f"after 3 attempts: {last_error}"
            )
            # Use a minimal fallback stub
            final_stub = f'"""\nStub for {clean_name} — auto-generated fallback.\n"""\npass\n'

        # Save stub
        stubs_dict[clean_name] = final_stub

        stub_save_name = clean_name.replace("/", "_")
        stub_path = os.path.join(stubs_dir, stub_save_name + ".py")
        os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
        with open(stub_path, "w", encoding="utf-8") as f:
            f.write(final_stub)
            f.write("\n")

        # Save artifact
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
        sigs = extract_stub_signatures(final_stub)
        logger.info(f"    Signatures: {list(sigs.keys())}")

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_cost)

    # ---------- Save combined stubs for reference ----------
    combined_path = os.path.join(output_dir, "interface_stubs_combined.py")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("# Combined Interface Stubs (auto-generated)\n")
        f.write("# Each section is the verified stub for one file.\n\n")
        for fname in file_order:
            clean = sanitize_todo_file_name(fname)
            if clean in stubs_dict:
                f.write(f"\n# {'='*60}\n")
                f.write(f"# {clean}\n")
                f.write(f"# {'='*60}\n\n")
                f.write(stubs_dict[clean])
                f.write("\n\n")

    # Save stubs index
    index_path = os.path.join(stubs_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "file_order": file_order,
            "stubs": {k: extract_stub_signatures(v) for k, v in stubs_dict.items()},
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
