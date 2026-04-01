"""Baseline Agent: generates a complete, runnable experimental baseline with placeholder model."""

import argparse
import os
import shutil
import sys

from core.exceptions import PipelineError
from core.logger import setup_logging, get_logger
from core.parser.pdf_process import run_pdf_process
from workflow.planning import run_planning
from workflow.extracting_artifacts import run_extracting_artifacts
from workflow.analyzing import run_analyzing
from workflow.api_predefine import run_api_predefine
from workflow.coding import run_coding

logger = get_logger(__name__)

PROMPT_SET = "baseline"

DEFAULT_STAGES = ["preprocess", "planning", "extract", "analyzing", "api_predefine", "coding"]


def run_baseline_pipeline(paper_name: str,
                          gpt_version: str,
                          output_dir: str,
                          output_repo_dir: str,
                          paper_format: str = "JSON",
                          pdf_json_path: str = None,
                          pdf_latex_path: str = None,
                          stages: list = None) -> None:
    """Step 1: Generate runnable baseline experiment code."""

    if stages is None:
        stages = list(DEFAULT_STAGES)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_repo_dir, exist_ok=True)

    cleaned_json_path = None
    if pdf_json_path:
        cleaned_json_path = pdf_json_path.replace('.json', '_cleaned.json')

    if "preprocess" in stages:
        logger.info("------- [Baseline] Preprocess -------")
        run_pdf_process(pdf_json_path, cleaned_json_path)

    if "planning" in stages:
        logger.info("------- [Baseline] Planning -------")
        run_planning(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    if "extract" in stages:
        logger.info("------- [Baseline] Extract Artifacts -------")
        run_extracting_artifacts(output_dir=output_dir)
        shutil.copy(
            f"{output_dir}/planning_config.yaml",
            f"{output_repo_dir}/config.yaml",
        )

    if "analyzing" in stages:
        logger.info("------- [Baseline] Analyzing -------")
        run_analyzing(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    if "api_predefine" in stages:
        logger.info("------- [Baseline] API Predefine -------")
        run_api_predefine(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    if "coding" in stages:
        logger.info("------- [Baseline] Coding -------")
        run_coding(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            output_repo_dir=output_repo_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
        )

    logger.info("------- [Baseline] Done -------")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Baseline Agent Pipeline")
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--gpt_version', type=str, default="gpt-5-mini")
    parser.add_argument('--paper_format', type=str, default="JSON",
                        choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_repo_dir', type=str, required=True)
    parser.add_argument('--stages', type=str, nargs='+', default=DEFAULT_STAGES)
    args = parser.parse_args()

    try:
        run_baseline_pipeline(
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
