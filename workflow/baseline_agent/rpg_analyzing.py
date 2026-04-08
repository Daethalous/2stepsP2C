"""
RPG-Enhanced Analyzing -- Phase 3a of the RPG integration.

Enhances the analyzing stage with:
  1. RPG-sorted file order (dependencies analyzed first)
  2. Upstream dependency context injected into analysis prompts
     (so the LLM knows what APIs are available from dependency files)

This is a drop-in replacement for workflow.analyzing.run_analyzing().
"""

import argparse
import copy
import json
import os
from tqdm import tqdm

from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.llm_engine import chat_completion_with_retry, create_client
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.utils import (
    extract_interface_signatures,
    format_paper_content_for_prompt,
    get_injection_info_for_file,
    load_accumulated_cost,
    parse_feature_design,
    parse_structured_json,
    print_log_cost,
    print_response,
    save_accumulated_cost,
    validate_required_keys,
)

from workflow.baseline_agent.build_rpg import PipelineRPG, build_rpg_from_planning
from workflow.baseline_agent.rpg_adapter import get_analysis_context

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def _load_or_build_rpg(output_dir: str) -> PipelineRPG:
    """Load existing RPG or build from planning output."""
    rpg_path = os.path.join(output_dir, "rpg_graph.json")
    if os.path.exists(rpg_path):
        logger.info(f"  [RPG] Loading existing RPG from {rpg_path}")
        return PipelineRPG.load(rpg_path)
    else:
        logger.info(f"  [RPG] Building RPG from planning output")
        rpg = build_rpg_from_planning(output_dir)
        rpg.save(rpg_path)
        return rpg


def run_rpg_analyzing(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    prompt_set: str = None,
    baseline_repo_dir: str = None,
    live_repo_dir: str = None,
) -> None:
    """
    RPG-enhanced analyzing stage.

    Differences from the original:
      1. Files are analyzed in RPG topological order
      2. Each file's analysis prompt includes upstream dependency context
         (logic analysis of files it depends on)
      3. Saves the RPG if not already saved
    """
    logger.info("=== RPG-Enhanced Analyzing Stage ===")

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=22000)

    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    context_lst = ctx.context_lst
    overview_prompt = format_paper_content_for_prompt(context_lst[0], max_chars=12000) if len(context_lst) > 0 else ""
    design_prompt = format_paper_content_for_prompt(context_lst[1], max_chars=16000) if len(context_lst) > 1 else ""
    task_prompt = format_paper_content_for_prompt(context_lst[2], max_chars=16000) if len(context_lst) > 2 else ""

    logic_analysis_dict = dict(ctx.logic_analysis_dict)

    # ---------- Load/Build RPG ----------
    rpg = _load_or_build_rpg(output_dir)

    # Use RPG-sorted order
    original_order = ctx.todo_file_lst
    rpg_sorted_order = rpg.topological_sort()
    todo_file_lst = [f for f in rpg_sorted_order if f in set(original_order)]

    # Add any files from original that RPG might have missed
    rpg_set = set(todo_file_lst)
    for f in original_order:
        if f not in rpg_set:
            todo_file_lst.append(f)

    logger.info(f"  File order: RPG-sorted ({len(todo_file_lst)} files)")

    injection_points = []
    if prompt_set == "feature":
        injection_points, _, _ = parse_feature_design(context_lst[1])

    done_file_lst = ["config.yaml"]

    # Track completed analysis for upstream context
    completed_analysis_dict = {}

    analysis_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_system.txt"),
            paper_format=paper_format)}
    ]

    def _load_feature_file_code(todo_file_name: str):
        candidates = [todo_file_name]
        clean_name = sanitize_todo_file_name(todo_file_name)
        if clean_name and clean_name not in candidates:
            candidates.append(clean_name)
        if prompt_set == "feature" and live_repo_dir:
            for candidate in candidates:
                live_path = os.path.join(live_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live"
        if prompt_set == "feature" and baseline_repo_dir:
            for candidate in candidates:
                baseline_path = os.path.join(baseline_repo_dir, candidate)
                if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as bf:
                        return bf.read(), "baseline"
        return None, "new"

    def get_write_msg(todo_file_name, todo_file_desc):
        draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
        if len(todo_file_desc.strip()) == 0:
            draft_desc = f"Write the logic analysis in '{todo_file_name}'."

        # NEW: Add upstream dependency context from RPG
        dep_context = get_analysis_context(rpg, todo_file_name, completed_analysis_dict, max_chars=4000)
        if dep_context:
            deps = rpg.get_dependencies(todo_file_name)
            draft_desc += (
                f"\n\n## Upstream Dependencies\n"
                f"This file depends on: {deps}\n"
                f"Below are the logic analyses of these dependency files. "
                f"Your analysis should be consistent with their interfaces.\n\n"
                f"{dep_context}"
            )

        extra_kwargs = {}
        if prompt_set == "feature" and baseline_repo_dir:
            file_code, code_source = _load_feature_file_code(todo_file_name)
            if file_code is None:
                extra_kwargs["baseline_file_code"] = "(new file -- no baseline/live code)"
            else:
                sig = extract_interface_signatures(file_code, max_lines=120)
                compact_code = file_code[:14000]
                if len(file_code) > 14000:
                    compact_code += "\n...(truncated for token budget)..."
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {code_source}\n"
                    "### Interface Signatures\n"
                    f"{sig}\n\n"
                    "### Code Snapshot\n"
                    f"{compact_code}"
                )
            extra_kwargs["injection_info"] = get_injection_info_for_file(
                injection_points, todo_file_name)

        write_msg = [{"role": "user", "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_user.txt"),
            paper_content=paper_content_prompt,
            overview=overview_prompt,
            design=design_prompt,
            task=task_prompt,
            config_yaml=config_yaml,
            draft_desc=draft_desc,
            todo_file_name=todo_file_name,
            **extra_kwargs)}]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    # ---------- Main analysis loop ----------
    artifact_output_dir = f"{output_dir}/analyzing_artifacts"
    os.makedirs(artifact_output_dir, exist_ok=True)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")

    for todo_file_name in tqdm(todo_file_lst):
        responses = []
        trajectories = copy.deepcopy(analysis_msg)

        current_stage = f"[RPG_ANALYSIS] {todo_file_name}"
        logger.info(current_stage)

        if todo_file_name == "config.yaml":
            continue

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        _skip_name = clean_todo_file_name.replace("/", "_")
        _skip_path = os.path.join(artifact_output_dir, f"{_skip_name}_simple_analysis.txt")
        if os.path.exists(_skip_path):
            logger.info(f"  [SKIP] artifact already exists: {_skip_path}")
            done_file_lst.append(clean_todo_file_name)
            # Load existing analysis for upstream context
            with open(_skip_path, "r", encoding="utf-8") as f:
                completed_analysis_dict[clean_todo_file_name] = f.read()
            continue

        # Log RPG dependency info
        deps = rpg.get_dependencies(clean_todo_file_name)
        if deps:
            analyzed = [d for d in deps if d in completed_analysis_dict]
            pending = [d for d in deps if d not in completed_analysis_dict]
            logger.info(f"  [RPG] Deps: analyzed={analyzed}, pending={pending}")

        if clean_todo_file_name not in logic_analysis_dict:
            logic_analysis_dict[clean_todo_file_name] = ""

        instruction_msg = get_write_msg(clean_todo_file_name, logic_analysis_dict[clean_todo_file_name])
        trajectories.extend(instruction_msg)

        completion = None
        completion_json = None
        for attempt in range(3):
            completion = api_call(trajectories)
            completion_json = json.loads(completion.model_dump_json())
            if prompt_set != "feature":
                break
            model_text = completion_json["choices"][0]["message"]["content"]
            payload = parse_structured_json(model_text)
            required = [
                "file", "modification_steps", "interface_contract_checklist",
                "config_keys_used", "test_focus", "blocked_items",
            ]
            if validate_required_keys(payload, required):
                break
            trajectories.append({"role": "assistant", "content": model_text})
            trajectories.append({
                "role": "user",
                "content": (
                    "Your output format is invalid. Return ONLY [CONTENT]{json}[/CONTENT] "
                    f"with required keys: {required}. No extra text."
                ),
            })
            logger.warning(f"  [RETRY {attempt+1}/3] Schema invalid for {clean_todo_file_name}")

        responses.append(completion_json)
        message = completion.choices[0].message
        trajectories.append({"role": message.role, "content": message.content})

        # Store completed analysis for downstream files
        analysis_text = completion_json["choices"][0]["message"]["content"]
        completed_analysis_dict[clean_todo_file_name] = analysis_text

        print_response(completion_json)
        total_accumulated_cost = print_log_cost(
            completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost
        )

        save_todo_file_name = clean_todo_file_name.replace("/", "_")
        artifact_file_path = os.path.join(artifact_output_dir, f"{save_todo_file_name}_simple_analysis.txt")
        os.makedirs(os.path.dirname(artifact_file_path), exist_ok=True)
        with open(artifact_file_path, "w", encoding="utf-8") as f:
            f.write(analysis_text)

        done_file_lst.append(clean_todo_file_name)

        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_response.json"), "w", encoding="utf-8") as f:
            json.dump(responses, f)
        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_trajectories.json"), "w", encoding="utf-8") as f:
            json.dump(trajectories, f)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
    logger.info("=== RPG-Enhanced Analyzing Complete ===")
