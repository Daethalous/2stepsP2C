"""Baseline Agent: delegates baseline generation to the RPG workflow."""

import argparse
import sys

from core.exceptions import PipelineError
from core.logger import setup_logging, get_logger
from workflow.baseline_agent.rpg_pipeline import (
    DEFAULT_STAGES,
    run_rpg_baseline_pipeline,
)

logger = get_logger(__name__)

def run_baseline_pipeline(paper_name: str,
                          gpt_version: str,
                          output_dir: str,
                          output_repo_dir: str,
                          paper_format: str = "JSON",
                          pdf_json_path: str = None,
                          pdf_latex_path: str = None,
                          stages: list = None) -> None:
    """Step 1: Generate baseline experiment code through the RPG workflow."""
    run_rpg_baseline_pipeline(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        paper_format=paper_format,
        pdf_json_path=pdf_json_path,
        pdf_latex_path=pdf_latex_path,
        stages=stages,
    )


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
