import argparse
import os
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval
from pipeline.feature_agent import run_feature_pipeline


def _cleanup_outputs(output_dir: str, output_repo_dir: str) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(output_repo_dir):
        shutil.rmtree(output_repo_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_repo_dir, exist_ok=True)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, default="iTransformer")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join("data", "paper2code", "paper2code_data", "iclr2024", "iTransformer.json"),
    )
    parser.add_argument(
        "--baseline_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round_repo"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_ref_eval"),
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_ref_eval_repo"),
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "iTransformer"),
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument(
        "--feature_stages",
        type=str,
        nargs="+",
        default=["planning", "extract", "analyzing", "coding"],
    )
    parser.add_argument("--skip_cleanup", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    if not args.skip_cleanup:
        _cleanup_outputs(args.output_dir, args.output_repo_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.output_repo_dir, exist_ok=True)

    cleaned_json_path = args.pdf_json_path.replace(".json", "_cleaned.json")
    eval_result_dir = os.path.join(args.output_dir, "eval_results")

    if not args.skip_feature:
        run_feature_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            baseline_repo_dir=args.baseline_repo_dir,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            stages=args.feature_stages,
        )

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
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
            output_dir=args.output_dir,
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
    print(f"[DONE] baseline_repo_dir: {args.baseline_repo_dir}")
    print(f"[DONE] feature_output_dir: {args.output_dir}")
    print(f"[DONE] feature_output_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] eval_result_dir: {eval_result_dir}")


if __name__ == "__main__":
    setup_logging()
    main()
