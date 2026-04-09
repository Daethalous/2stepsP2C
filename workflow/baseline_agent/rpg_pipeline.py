"""
RPG-Enhanced Baseline Agent Pipeline.

Drop-in replacement for pipeline/baseline_agent.py that uses
RPG graph-based planning for dependency-aware file ordering,
focused coding context, and cross-file signature validation.

New stage added:
  - build_rpg (Stage 3.5): Builds dependency graph from planning output

Enhanced stages:
  - analyzing:    RPG-sorted order + upstream dependency context
  - api_predefine: Cross-module dependency info in prompt
  - coding:       Graph-aware context + signature validation + auto-retry

Usage:
  python -m workflow.baseline_agent.rpg_pipeline \
    --paper_name "iTransformer" \
    --gpt_version "gpt-5-mini" \
    --paper_format JSON \
    --pdf_json_path "path/to/paper_cleaned.json" \
    --output_dir "outputs/iTransformer_artifacts" \
    --output_repo_dir "outputs/iTransformer_repo"

  # Run only specific stages:
  python -m workflow.baseline_agent.rpg_pipeline \
    --paper_name "iTransformer" \
    --output_dir "outputs/iTransformer_artifacts" \
    --output_repo_dir "outputs/iTransformer_repo" \
    --stages build_rpg analyzing api_predefine coding
"""

import argparse
import os
import shutil
import sys

from core.exceptions import PipelineError
from core.logger import setup_logging, get_logger

logger = get_logger(__name__)

PROMPT_SET = "baseline"

DEFAULT_STAGES = [
    "preprocess",
    "planning",
    "extract",
    "build_rpg",           # Stage 3.5: Build dependency graph
    "analyzing",           # Enhanced with RPG
    "interface_design",    # NEW: Per-file stub generation (replaces api_predefine)
    "coding",              # Enhanced with stub enforcement
    "typecheck",           # NEW: Optional pyright type checking
]


def run_rpg_baseline_pipeline(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    output_repo_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    stages: list = None,
) -> None:
    """
    RPG-enhanced baseline pipeline.

    Stages 1-3 (preprocess, planning, extract) are identical to the original.
    Stage 3.5 (build_rpg) is new.
    Stages 4-6 (analyzing, api_predefine, coding) use RPG-enhanced versions.
    """
    if stages is None:
        stages = list(DEFAULT_STAGES)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_repo_dir, exist_ok=True)

    cleaned_json_path = None
    if pdf_json_path:
        cleaned_json_path = pdf_json_path.replace(".json", "_cleaned.json")

    # ─── Stage 1: Preprocess ───
    if "preprocess" in stages:
        logger.info("------- [RPG Baseline] Preprocess -------")
        from core.parser.pdf_process import run_pdf_process
        run_pdf_process(pdf_json_path, cleaned_json_path)

    # ─── Stage 2: Planning ───
    if "planning" in stages:
        logger.info("------- [RPG Baseline] Planning -------")
        from workflow.planning import run_planning
        run_planning(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    # ─── Stage 3: Extract Artifacts ───
    if "extract" in stages:
        logger.info("------- [RPG Baseline] Extract Artifacts -------")
        from workflow.extracting_artifacts import run_extracting_artifacts
        run_extracting_artifacts(output_dir=output_dir)
        shutil.copy(
            f"{output_dir}/planning_config.yaml",
            f"{output_repo_dir}/config.yaml",
        )

    # ─── Stage 3.5: Build RPG ─── (NEW)
    if "build_rpg" in stages:
        logger.info("------- [RPG Baseline] Build RPG (Stage 3.5) -------")
        from workflow.baseline_agent.build_rpg import (
            build_rpg_from_planning,
            compare_file_orders,
            print_dependency_graph,
        )
        from core.data_loader import load_pipeline_context

        rpg = build_rpg_from_planning(output_dir, gpt_version=gpt_version)
        rpg_path = os.path.join(output_dir, "rpg_graph.json")
        rpg.save(rpg_path)

        # Log the dependency graph and file order comparison
        ctx = load_pipeline_context(output_dir)
        sorted_files = rpg.topological_sort()
        comparison = compare_file_orders(ctx.todo_file_lst, sorted_files)
        logger.info(comparison)

        # Save sorted order
        import json
        sorted_path = os.path.join(output_dir, "rpg_sorted_files.json")
        with open(sorted_path, "w", encoding="utf-8") as f:
            json.dump({
                "original_order": ctx.todo_file_lst,
                "rpg_sorted_order": sorted_files,
            }, f, indent=2)
        logger.info(f"  RPG saved to {rpg_path}")
        logger.info(f"  Sorted order saved to {sorted_path}")

    # ─── Stage 4: Analyzing (RPG-enhanced) ───
    if "analyzing" in stages:
        logger.info("------- [RPG Baseline] Analyzing (RPG-enhanced) -------")
        from workflow.baseline_agent.rpg_analyzing import run_rpg_analyzing
        run_rpg_analyzing(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    # ─── Stage 5: Interface Design (replaces api_predefine) ───
    if "interface_design" in stages:
        logger.info("------- [RPG Baseline] Interface Design (Per-File Stubs) -------")
        from workflow.baseline_agent.rpg_interface_design import run_interface_design
        run_interface_design(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    # ─── Legacy: API Predefine (kept for backward compatibility) ───
    if "api_predefine" in stages:
        logger.info("------- [RPG Baseline] API Predefine (Legacy) -------")
        from workflow.baseline_agent.rpg_api_predefine import run_rpg_api_predefine
        run_rpg_api_predefine(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    # ─── Stage 7: Coding (RPG-enhanced) ───
    if "coding" in stages:
        logger.info("------- [RPG Baseline] Coding (RPG-enhanced) -------")
        from workflow.baseline_agent.rpg_coding import run_rpg_coding
        run_rpg_coding(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            output_repo_dir=output_repo_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    # ─── Stage 8: Type Check (Optional — requires pyright) ───
    if "typecheck" in stages:
        logger.info("------- [RPG Baseline] Type Check (Pyright) -------")
        from workflow.baseline_agent.rpg_typecheck import run_typecheck
        success, report, errors = run_typecheck(
            repo_dir=output_repo_dir,
            output_dir=output_dir,
        )
        if success and errors:
            error_count = sum(1 for e in errors if e["severity"] == "error")
            if error_count > 0:
                logger.warning(f"  [TYPECHECK] {error_count} type errors found. See pyright_report.txt")

    logger.info("------- [RPG Baseline] Done -------")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="RPG-Enhanced Baseline Agent Pipeline"
    )
    parser.add_argument("--paper_name", type=str, required=True)
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--paper_format", type=str, default="JSON",
                        choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str)
    parser.add_argument("--pdf_latex_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_repo_dir", type=str, required=True)
    parser.add_argument("--stages", type=str, nargs="+", default=DEFAULT_STAGES,
                        help=f"Stages to run. Default: {DEFAULT_STAGES}")
    args = parser.parse_args()

    try:
        run_rpg_baseline_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            paper_format=args.paper_format,
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=args.pdf_latex_path,
            stages=args.stages,
        )
    except PipelineError as e:
        logger.error(f"[ERROR] {e}")
        sys.exit(1)
