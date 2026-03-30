import argparse
import os
import sys
import shutil
import json
import time

from core.exceptions import PipelineError
from core.logger import setup_logging, get_logger
from core.parser.pdf_process import run_pdf_process
from workflow.planning import run_planning
from workflow.extracting_artifacts import run_extracting_artifacts
from workflow.analyzing import run_analyzing
from workflow.coding import run_coding
from evaluation.eval import run_eval
from pipeline.baseline_agent import run_baseline_pipeline
from pipeline.feature_agent import run_feature_pipeline

logger = get_logger(__name__)
DEBUG_LOG_PATH = "debug-084f81.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "084f81",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # #endregion


def _run_single_mode(args, cleaned_json_path):
    """Original single-step pipeline (prompt_set=None, backward compatible)."""
    if "preprocess" in args.stages:
        logger.info("------- Preprocess -------")
        run_pdf_process(args.pdf_json_path, cleaned_json_path)

    if "planning" in args.stages:
        logger.info("------- Planning -------")
        run_planning(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            paper_format=args.paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=args.pdf_latex_path,
        )

    if "extract" in args.stages:
        logger.info("------- Extract Artifacts -------")
        run_extracting_artifacts(output_dir=args.output_dir)
        shutil.copy(
            f"{args.output_dir}/planning_config.yaml",
            f"{args.output_repo_dir}/config.yaml",
        )

    if "analyzing" in args.stages:
        logger.info("------- Analyzing -------")
        run_analyzing(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            paper_format=args.paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=args.pdf_latex_path,
        )

    if "coding" in args.stages:
        logger.info("------- Coding -------")
        run_coding(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            paper_format=args.paper_format,
            pdf_json_path=cleaned_json_path,
            pdf_latex_path=args.pdf_latex_path,
        )

    _run_eval(args, cleaned_json_path, args.output_repo_dir, args.output_dir)


def _run_baseline_mode(args, cleaned_json_path):
    """Run only the Baseline Agent (Step 1)."""
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
    _run_eval(args, cleaned_json_path, args.output_repo_dir, args.output_dir)


def _run_feature_mode(args, cleaned_json_path):
    """Run only the Feature Agent (Step 2). Requires --baseline_repo_dir."""
    if not args.baseline_repo_dir:
        raise PipelineError("--baseline_repo_dir is required for feature mode.")

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
    _run_eval(args, cleaned_json_path, args.output_repo_dir, args.output_dir)


def _run_two_step_mode(args, cleaned_json_path):
    """Run the full two-step pipeline: Baseline Agent -> Feature Agent -> Eval."""
    base = args.output_dir
    repo_dir = args.output_repo_dir if args.output_repo_dir else f"{base}_repo"
    _debug_log(
        run_id="initial",
        hypothesis_id="H1",
        location="main.py:_run_two_step_mode",
        message="enter two_step",
        data={
            "file": __file__,
            "base": base,
            "repo_dir": repo_dir,
            "stages": args.stages,
        },
    )

    baseline_output_dir = f"{base}/baseline"
    baseline_snapshot_dir = f"{base}/baseline_repo"
    feature_output_dir = f"{base}/feature"

    os.makedirs(repo_dir, exist_ok=True)

    if "preprocess" in args.stages:
        logger.info("======= [Two-Step] Shared Preprocess =======")
        run_pdf_process(args.pdf_json_path, cleaned_json_path)

    baseline_stages = [s for s in args.stages
                       if s in ("planning", "extract", "analyzing", "coding")]
    if baseline_stages:
        logger.info("======= [Two-Step] Step 1: Baseline Agent =======")
        run_baseline_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=baseline_output_dir,
            output_repo_dir=repo_dir,
            paper_format=args.paper_format,
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=args.pdf_latex_path,
            stages=baseline_stages,
        )

    if os.path.exists(repo_dir) and baseline_stages:
        logger.info("======= [Two-Step] Snapshot baseline repo =======")
        if os.path.exists(baseline_snapshot_dir):
            shutil.rmtree(baseline_snapshot_dir)
        shutil.copytree(repo_dir, baseline_snapshot_dir)

    feature_stages = [s for s in args.stages
                      if s in ("planning", "extract", "analyzing", "coding")]
    if feature_stages:
        logger.info("======= [Two-Step] Step 2: Feature Agent =======")
        run_feature_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            output_repo_dir=repo_dir,
            baseline_repo_dir=baseline_snapshot_dir,
            paper_format=args.paper_format,
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=args.pdf_latex_path,
            stages=feature_stages,
        )

    _run_eval(args, cleaned_json_path, repo_dir, base,
              planning_dir=feature_output_dir)


def _run_eval(args, cleaned_json_path, target_repo_dir, eval_base_dir,
              planning_dir=None):
    """Run evaluation stages if requested.

    planning_dir: directory containing planning_config.yaml and planning_trajectories.json.
                  Defaults to eval_base_dir if not specified.
    """
    if planning_dir is None:
        planning_dir = eval_base_dir
    eval_result_dir = f"{eval_base_dir}/eval_results"

    if "eval_ref_free" in args.stages:
        logger.info("------- Evaluation (ref-free) -------")
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=planning_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=target_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    if "eval_ref_based" in args.stages:
        logger.info("------- Evaluation (ref-based) -------")
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=planning_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=target_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="PaperCoder Pipeline")
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--gpt_version', type=str, default="gpt-5-mini")
    parser.add_argument('--paper_format', type=str, default="JSON",
                        choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str,
                        help="Path to paper JSON (S2ORC format)")
    parser.add_argument('--pdf_latex_path', type=str,
                        help="Path to paper LaTeX source")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_repo_dir', type=str, default="",
                        help="Output repo dir (used in single/baseline/feature mode)")
    parser.add_argument('--gold_repo_dir', type=str, default="",
                        help="Path to gold repository for ref-based eval")
    parser.add_argument('--generated_n', type=int, default=8,
                        help="Number of eval samples to generate")
    parser.add_argument('--agent_mode', type=str, default="two_step",
                        choices=["single", "baseline", "feature", "two_step"],
                        help="Pipeline mode: single (original), baseline, feature, or two_step")
    parser.add_argument('--baseline_repo_dir', type=str, default="",
                        help="Baseline repo path (required for feature mode)")
    parser.add_argument('--stages', type=str, nargs='+',
                        default=["preprocess", "planning", "extract",
                                 "analyzing", "coding",
                                 "eval_ref_free", "eval_ref_based"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_repo_dir:
        os.makedirs(args.output_repo_dir, exist_ok=True)

    cleaned_json_path = None
    if args.pdf_json_path:
        cleaned_json_path = args.pdf_json_path.replace('.json', '_cleaned.json')

    logger.info(f"Agent mode: {args.agent_mode}")
    logger.info(f"Stages: {args.stages}")

    if args.agent_mode == "single":
        _run_single_mode(args, cleaned_json_path)
    elif args.agent_mode == "baseline":
        _run_baseline_mode(args, cleaned_json_path)
    elif args.agent_mode == "feature":
        _run_feature_mode(args, cleaned_json_path)
    elif args.agent_mode == "two_step":
        _run_two_step_mode(args, cleaned_json_path)

    logger.info("======= All Done =======")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except PipelineError as e:
        logger.error(f"[ERROR] {e}")
        sys.exit(1)
