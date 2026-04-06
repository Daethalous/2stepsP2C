import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval
from scripts._run_path_utils import make_run_tag, resolve_default_output_path


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
    parser.add_argument("--paper_name", type=str, default="RECOMBINER")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--target_repo_dir",
        type=str,
        default=os.path.join("outputs", "RECOMBINER_manual_repro_repo"),
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "RECOMBINER.json"
        ),
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "RECOMBINER"),
    )
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        default=os.path.join("outputs", "RECOMBINER_manual_repro_eval_results"),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix for eval output path. Defaults to a current timestamp when using the default path.",
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    default_eval_result_dir = os.path.join("outputs", "RECOMBINER_manual_repro_eval_results")
    run_tag = args.run_tag or make_run_tag()
    args.eval_result_dir = resolve_default_output_path(
        args.eval_result_dir,
        default_eval_result_dir,
        run_tag,
    )

    _require_dir(args.target_repo_dir, "target_repo_dir")
    _require_dir(args.gold_repo_dir, "gold_repo_dir")
    _require_file(args.pdf_json_path, "pdf_json_path")

    os.makedirs(args.eval_result_dir, exist_ok=True)

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.eval_result_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.target_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=False,
        )

    if not args.skip_eval_ref_based:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.eval_result_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.target_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=False,
        )


if __name__ == "__main__":
    setup_logging()
    main()
