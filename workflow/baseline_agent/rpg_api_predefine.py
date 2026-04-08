"""
RPG-Enhanced API Predefine -- Phase 3b of the RPG integration.

Enhances the api_predefine stage by injecting RPG cross-module
dependency information into the prompt, so the LLM knows which
APIs actually cross module boundaries and need to be consistent.

This is a drop-in replacement for workflow.api_predefine.run_api_predefine().
"""

import ast
import copy
import json
import os
import re
from typing import Dict, List

from core.data_loader import load_paper_content, load_pipeline_context
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
from workflow.baseline_agent.rpg_adapter import get_cross_module_interfaces

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def _extract_message_content(response_json_obj):
    try:
        return response_json_obj[0]["choices"][0]["message"]["content"]
    except Exception:
        return ""


def _extract_logic_core(text: str) -> str:
    """Extract interface-relevant parts from analysis text."""
    if not isinstance(text, str):
        return ""

    content = text
    block_match = re.search(r"\[CONTENT\](.*?)\[/CONTENT\]", content, re.DOTALL)
    if block_match:
        content = block_match.group(1)

    logic_match = re.search(
        r"(##\s*Logic Analysis[\s\S]*?)(?:\n##\s+|\Z)",
        content,
        re.IGNORECASE,
    )
    if logic_match:
        content = logic_match.group(1)

    filtered_lines = []
    keep_keywords = (
        "def ", "class ", "signature", "interface", "api",
        "parameter", "return", "input", "output", "shape",
        "dtype", "tensor", "function", "method",
    )
    skip_keywords = (
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cached_tokens", "usage", "http", "status_code",
        "request_id", "gpt-",
    )

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(k in low for k in skip_keywords):
            continue
        if line.startswith("```") or line.startswith("{") or line.startswith("}"):
            continue
        if any(k in low for k in keep_keywords):
            filtered_lines.append(line)
            continue
        if line.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.")):
            filtered_lines.append(line)

    if not filtered_lines:
        compact = re.sub(r"\s+", " ", content).strip()
        return compact[:1800]

    return "\n".join(filtered_lines[:80])


def _normalize_stub_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    txt = raw.strip()
    if txt.lower().startswith("python\n"):
        txt = txt.split("\n", 1)[1]
    return txt.strip()


def _validate_stub(code_str: str):
    if "# [GLOBAL API CONTRACT]" not in code_str:
        return False, "Missing required header: # [GLOBAL API CONTRACT]"
    low = code_str.lower()
    disallowed_tokens = ("notimplementederror", "# todo", "todo:")
    if any(token in low for token in disallowed_tokens):
        return False, "Contains disallowed tokens (TODO/NotImplementedError)"
    if ("def " not in code_str) and ("class " not in code_str):
        return False, "Stub must contain at least one class or def signature"
    try:
        ast.parse(code_str)
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}"
    return True, ""


def _load_or_build_rpg(output_dir: str) -> PipelineRPG:
    """Load existing RPG or build from planning output."""
    rpg_path = os.path.join(output_dir, "rpg_graph.json")
    if os.path.exists(rpg_path):
        return PipelineRPG.load(rpg_path)
    else:
        rpg = build_rpg_from_planning(output_dir)
        rpg.save(rpg_path)
        return rpg


def _build_dependency_order_summary(rpg: PipelineRPG) -> str:
    """Build a summary showing the dependency order and relationships."""
    sorted_files = rpg.topological_sort()
    lines = ["## File Dependency Order (topological)"]
    for i, f in enumerate(sorted_files):
        deps = rpg.get_dependencies(f)
        if deps:
            lines.append(f"  {i+1}. {f} <- depends on: {', '.join(deps)}")
        else:
            lines.append(f"  {i+1}. {f} (no dependencies)")
    return "\n".join(lines)


def run_rpg_api_predefine(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    prompt_set: str = None,
) -> None:
    """
    RPG-enhanced API predefine stage.

    Differences from the original:
      1. Injects cross-module dependency info from RPG
      2. Includes dependency order summary
      3. Helps the LLM focus on APIs that actually cross file boundaries
    """
    logger.info("=== RPG-Enhanced API Predefine Stage ===")

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=22000)

    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    context_lst = ctx.context_lst
    overview_prompt = format_paper_content_for_prompt(context_lst[0], max_chars=12000) if len(context_lst) > 0 else ""
    design_prompt = format_paper_content_for_prompt(context_lst[1], max_chars=16000) if len(context_lst) > 1 else ""
    task_prompt = format_paper_content_for_prompt(context_lst[2], max_chars=16000) if len(context_lst) > 2 else ""
    todo_file_lst = ctx.todo_file_lst

    # ---------- Load RPG and build cross-module info ----------
    rpg = _load_or_build_rpg(output_dir)
    cross_module_info = get_cross_module_interfaces(rpg)
    dep_order_summary = _build_dependency_order_summary(rpg)

    logger.info(f"  [RPG] Cross-module dependencies:\n{cross_module_info}")

    # ---------- Gather analysis summaries ----------
    analysis_summaries = []
    for todo_file_name in todo_file_lst:
        if todo_file_name == "config.yaml":
            continue
        save_name = todo_file_name.replace("/", "_")
        response_path = f"{output_dir}/{save_name}_simple_analysis_response.json"
        if not os.path.exists(response_path):
            continue
        with open(response_path, "r", encoding="utf-8") as f:
            response_json = json.load(f)
        text = _extract_message_content(response_json)
        core = _extract_logic_core(text)
        if core:
            analysis_summaries.append(f"# {todo_file_name}\n{core}")

    if not analysis_summaries:
        raise FileNotFoundError(
            "No analyzing summaries found. Run analyzing stage first."
        )

    # NEW: Append RPG dependency info to the analysis summaries
    analyzing_outputs_summary = "\n\n".join(analysis_summaries)
    analyzing_outputs_summary += (
        f"\n\n{dep_order_summary}"
        f"\n\n{cross_module_info}"
        f"\n\n## IMPORTANT: Cross-Module API Consistency\n"
        f"The dependency graph above shows which files import from which other files.\n"
        f"Your API contract MUST ensure that function signatures are consistent\n"
        f"across these cross-module boundaries. Pay special attention to:\n"
        f"- Parameter names and types must match between caller and callee\n"
        f"- Return types must match what the caller expects\n"
        f"- Tensor shapes must be consistent across the pipeline\n"
    )

    # ---------- Build prompt ----------
    stage_msg = [
        {
            "role": "system",
            "content": render_prompt(
                _prompt_path(prompt_set, "api_predefine_system.txt"),
                paper_format=paper_format,
            ),
        }
    ]

    instruction_msg = [
        {
            "role": "user",
            "content": render_prompt(
                _prompt_path(prompt_set, "api_predefine_user.txt"),
                paper_content=paper_content_prompt,
                overview=overview_prompt,
                design=design_prompt,
                task=task_prompt,
                config_yaml=config_yaml,
                todo_file_lst=todo_file_lst,
                analyzing_outputs_summary=analyzing_outputs_summary,
            ),
        }
    ]

    trajectories = copy.deepcopy(stage_msg)
    trajectories.extend(instruction_msg)
    responses = []

    completion = None
    final_stub = ""
    last_error = "Unknown validation error"

    for _ in range(3):
        completion = chat_completion_with_retry(client, gpt_version, trajectories)
        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)

        model_text = completion_json["choices"][0]["message"]["content"]
        stub_text = extract_code_from_content(model_text)
        if not stub_text:
            stub_text = model_text
        stub_text = _normalize_stub_text(stub_text)

        ok, err_msg = _validate_stub(stub_text)
        if ok:
            final_stub = stub_text
            last_error = ""
            break

        last_error = err_msg
        trajectories.append({"role": "assistant", "content": model_text})
        trajectories.append(
            {
                "role": "user",
                "content": (
                    "Your output is invalid.\n"
                    f"Validation error: {err_msg}\n"
                    "Return ONLY one Python code block containing a valid .pyi-style global API contract.\n"
                    "Keep the header '# [GLOBAL API CONTRACT]' and fix syntax issues.\n"
                    "Note: pass is allowed in stubs."
                ),
            }
        )

    if not final_stub:
        raise ValueError(f"api_predefine failed after retries: {last_error}")

    # ---------- Save artifacts ----------
    artifact_output_dir = f"{output_dir}/api_predefine_artifacts"
    os.makedirs(artifact_output_dir, exist_ok=True)

    contract_path = f"{output_dir}/api_predefine_contract.pyi"
    with open(contract_path, "w", encoding="utf-8") as f:
        f.write(final_stub)
        f.write("\n")

    with open(f"{artifact_output_dir}/2.5_global_api_contract.pyi", "w", encoding="utf-8") as f:
        f.write(final_stub)
        f.write("\n")

    with open(f"{output_dir}/api_predefine_response.json", "w", encoding="utf-8") as f:
        json.dump(responses, f)

    with open(f"{output_dir}/api_predefine_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(trajectories, f)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    last_completion_json = responses[-1]
    print_response(last_completion_json)
    total_accumulated_cost = print_log_cost(
        last_completion_json, gpt_version,
        "[RPG_API_PREDEFINE] global_stub",
        output_dir, total_accumulated_cost,
    )
    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

    logger.info(f"  [DONE] API contract saved: {contract_path}")
    logger.info("=== RPG-Enhanced API Predefine Complete ===")
