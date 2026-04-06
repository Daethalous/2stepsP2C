import argparse
import os
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval
from pipeline.baseline_agent import run_baseline_pipeline
from pipeline.feature_agent import run_feature_pipeline
from scripts._run_path_utils import make_run_tag, resolve_default_output_path


def _cleanup_dirs(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, default="RECOMBINER")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data",
            "paper2code",
            "paper2code_data",
            "iclr2024",
            "RECOMBINER.json",
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "RECOMBINER_full_repro_round"),
        help="Root directory for baseline/feature/ref-eval outputs",
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "RECOMBINER_full_repro_round_repo"),
        help="Live repo modified by feature stage and evaluated at the end",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix for output paths. Defaults to a current timestamp when using default output paths.",
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join(
            "data", "paper2code", "gold_repos", "RECOMBINER"
        ),
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument(
        "--baseline_stages",
        type=str,
        nargs="+",
        default=["preprocess", "planning", "extract", "analyzing", "api_predefine", "coding"],
    )
    parser.add_argument(
        "--feature_stages",
        type=str,
        nargs="+",
        default=["planning", "extract", "analyzing", "coding"],
    )
    parser.add_argument("--skip_cleanup", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    default_output_dir = os.path.join("outputs", "RECOMBINER_full_repro_round")
    default_output_repo_dir = os.path.join("outputs", "RECOMBINER_full_repro_round_repo")
    run_tag = args.run_tag or make_run_tag()
    args.output_dir = resolve_default_output_path(args.output_dir, default_output_dir, run_tag)
    args.output_repo_dir = resolve_default_output_path(
        args.output_repo_dir,
        default_output_repo_dir,
        run_tag,
    )

    baseline_output_dir = os.path.join(args.output_dir, "baseline")
    baseline_snapshot_dir = os.path.join(args.output_dir, "baseline_repo")
    feature_output_dir = os.path.join(args.output_dir, "feature")
    eval_result_dir = os.path.join(args.output_dir, "eval_results")

    if not args.skip_cleanup:
        _cleanup_dirs(
            baseline_output_dir,
            baseline_snapshot_dir,
            feature_output_dir,
            eval_result_dir,
            args.output_repo_dir,
        )
    else:
        os.makedirs(baseline_output_dir, exist_ok=True)
        os.makedirs(baseline_snapshot_dir, exist_ok=True)
        os.makedirs(feature_output_dir, exist_ok=True)
        os.makedirs(eval_result_dir, exist_ok=True)
        os.makedirs(args.output_repo_dir, exist_ok=True)

    cleaned_json_path = args.pdf_json_path.replace(".json", "_cleaned.json")

    if not args.skip_eval_ref_based and not os.path.isdir(args.gold_repo_dir):
        raise FileNotFoundError(f"gold_repo_dir not found: {args.gold_repo_dir}")

    if not args.skip_baseline:
        run_baseline_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=baseline_output_dir,
            output_repo_dir=args.output_repo_dir,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            stages=args.baseline_stages,
        )
        if os.path.exists(baseline_snapshot_dir):
            shutil.rmtree(baseline_snapshot_dir)
        shutil.copytree(args.output_repo_dir, baseline_snapshot_dir)

    if not args.skip_feature:
        run_feature_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            output_repo_dir=args.output_repo_dir,
            baseline_repo_dir=baseline_snapshot_dir,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            stages=args.feature_stages,
            api_predefine_contract_path=os.path.join(
                baseline_output_dir, "api_predefine_contract.pyi"
            ),
        )

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    if not args.skip_eval_ref_based:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] paper_name: {args.paper_name}")
    print(f"[DONE] gpt_version: {args.gpt_version}")
    print(f"[DONE] baseline_output_dir: {baseline_output_dir}")
    print(f"[DONE] feature_output_dir: {feature_output_dir}")
    print(f"[DONE] live_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] baseline_snapshot_dir: {baseline_snapshot_dir}")
    print(f"[DONE] eval_result_dir: {eval_result_dir}")
    print(f"[DONE] gold_repo_dir: {args.gold_repo_dir}")
    print(f"[DONE] run_tag: {run_tag}")


if __name__ == "__main__":
    setup_logging()
    main()
