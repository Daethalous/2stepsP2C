import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval


def _require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, default="iTransformer")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--planning_output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_full_repro_round", "feature"),
        help="Feature planning/analyzing artifact directory consumed by evaluation",
    )
    parser.add_argument(
        "--target_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_full_repro_round_repo"),
        help="Patched generated repo to be re-evaluated",
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "iTransformer_cleaned.json"
        ),
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "iTransformer"),
    )
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        default=os.path.join(
            "outputs", "iTransformer_full_repro_round", "eval_results_after_main_integration"
        ),
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    _require_dir(args.planning_output_dir, "planning_output_dir")
    _require_dir(args.target_repo_dir, "target_repo_dir")
    _require_dir(args.gold_repo_dir, "gold_repo_dir")
    _require_file(args.pdf_json_path, "pdf_json_path")
    _require_file(
        os.path.join(args.planning_output_dir, "planning_config.yaml"),
        "planning_config.yaml",
    )
    planning_response = os.path.join(args.planning_output_dir, "planning_response.json")
    planning_traj = os.path.join(args.planning_output_dir, "planning_trajectories.json")
    if not os.path.isfile(planning_response) and not os.path.isfile(planning_traj):
        raise FileNotFoundError(
            "Neither planning_response.json nor planning_trajectories.json exists under planning_output_dir"
        )

    os.makedirs(args.eval_result_dir, exist_ok=True)

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.planning_output_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.target_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    if not args.skip_eval_ref_based:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.planning_output_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.target_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] planning_output_dir: {args.planning_output_dir}")
    print(f"[DONE] target_repo_dir: {args.target_repo_dir}")
    print(f"[DONE] eval_result_dir: {args.eval_result_dir}")


if __name__ == "__main__":
    setup_logging()
    main()
