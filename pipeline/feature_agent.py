"""Feature Agent: injects a paper's novel algorithm into baseline code."""

import argparse
import os
import shutil
import sys

from core.exceptions import PipelineError
from core.logger import setup_logging, get_logger
from core.utils import merge_yaml_configs
from workflow.planning import run_planning
from workflow.extracting_artifacts import run_extracting_artifacts
from workflow.analyzing import run_analyzing
from workflow.coding import run_coding

logger = get_logger(__name__)

PROMPT_SET = "feature"

DEFAULT_STAGES = ["planning", "extract", "analyzing", "coding"]


def run_feature_pipeline(paper_name: str,
                         gpt_version: str,
                         output_dir: str,
                         output_repo_dir: str,
                         baseline_repo_dir: str,
                         paper_format: str = "JSON",
                         pdf_json_path: str = None,
                         pdf_latex_path: str = None,
                         stages: list = None) -> None:
    """Step 2: Inject paper's novel algorithm into baseline code."""

    if stages is None:
        stages = list(DEFAULT_STAGES)

    os.makedirs(output_dir, exist_ok=True)

    baseline_repo_dir = os.path.abspath(baseline_repo_dir)
    output_repo_dir = os.path.abspath(output_repo_dir)
    if not os.path.isdir(baseline_repo_dir):
        raise PipelineError(f"baseline_repo_dir not found or not a directory: {baseline_repo_dir}")

    if baseline_repo_dir == output_repo_dir:
        logger.info(f"[Feature][In-Place] live repo == baseline repo: {output_repo_dir}")
    elif not os.path.exists(output_repo_dir):
        logger.info(f"[Feature][In-Place] initializing live repo from baseline: {output_repo_dir}")
        shutil.copytree(baseline_repo_dir, output_repo_dir)
    elif not os.path.isdir(output_repo_dir):
        raise PipelineError(f"output_repo_dir exists but is not a directory: {output_repo_dir}")
    else:
        logger.info(f"[Feature][In-Place] using existing live repo: {output_repo_dir}")

    cleaned_json_path = None
    if pdf_json_path:
        cleaned_json_path = pdf_json_path.replace('.json', '_cleaned.json')

    if "planning" in stages:
        logger.info("------- [Feature] Planning -------")
        run_planning(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
            baseline_repo_dir=baseline_repo_dir,
        )

    if "extract" in stages:
        logger.info("------- [Feature] Extract Artifacts -------")
        run_extracting_artifacts(output_dir=output_dir)
        base_cfg = os.path.join(output_repo_dir, "config.yaml")
        overlay_cfg = os.path.join(output_dir, "planning_config.yaml")
        if os.path.exists(base_cfg) and os.path.exists(overlay_cfg):
            merge_yaml_configs(
                base_path=base_cfg,
                overlay_path=overlay_cfg,
                output_path=base_cfg,
            )
        else:
            logger.warning(
                "[Feature][In-Place] skip config merge due to missing config file(s): "
                f"base={base_cfg}, overlay={overlay_cfg}"
            )

    if "analyzing" in stages:
        logger.info("------- [Feature] Analyzing -------")
        run_analyzing(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
            baseline_repo_dir=baseline_repo_dir,
            live_repo_dir=output_repo_dir,
        )

    if "coding" in stages:
        logger.info("------- [Feature] Coding -------")
        run_coding(
            paper_name=paper_name,
            gpt_version=gpt_version,
            output_dir=output_dir,
            output_repo_dir=output_repo_dir,
            paper_format=paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=pdf_latex_path,
            prompt_set=PROMPT_SET,
            baseline_repo_dir=baseline_repo_dir,
            live_repo_dir=output_repo_dir,
        )

    logger.info("------- [Feature] Done -------")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Feature Agent Pipeline")
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--gpt_version', type=str, default="gpt-5-mini")
    parser.add_argument('--paper_format', type=str, default="JSON",
                        choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_repo_dir', type=str, required=True)
    parser.add_argument('--baseline_repo_dir', type=str, required=True,
                        help="Path to baseline repo from Step 1")
    parser.add_argument('--stages', type=str, nargs='+', default=DEFAULT_STAGES)
    args = parser.parse_args()

    try:
        run_feature_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            baseline_repo_dir=args.baseline_repo_dir,
            paper_format=args.paper_format,
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=args.pdf_latex_path,
            stages=args.stages,
        )
    except PipelineError as e:
        logger.error(f"[ERROR] {e}")
        sys.exit(1)
