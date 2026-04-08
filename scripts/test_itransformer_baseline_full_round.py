import argparse
import os
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from pipeline.baseline_agent import run_baseline_pipeline, DEFAULT_STAGES
from scripts._run_path_utils import make_run_tag, resolve_default_output_path


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
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round"),
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round_repo"),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix for output paths. Defaults to a current timestamp when using default output paths.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=list(DEFAULT_STAGES),
        help="Stages to run. Default: preprocess planning extract analyzing api_predefine coding",
    )
    parser.add_argument(
        "--skip_cleanup",
        action="store_true",
        help="Skip deleting existing output_dir/output_repo_dir before running",
    )
    args = parser.parse_args()

    default_output_dir = os.path.join("outputs", "iTransformer_baseline_full_round")
    default_output_repo_dir = os.path.join("outputs", "iTransformer_baseline_full_round_repo")
    run_tag = args.run_tag or make_run_tag()
    args.output_dir = resolve_default_output_path(args.output_dir, default_output_dir, run_tag)
    args.output_repo_dir = resolve_default_output_path(
        args.output_repo_dir,
        default_output_repo_dir,
        run_tag,
    )

    if not args.skip_cleanup:
        _cleanup_outputs(args.output_dir, args.output_repo_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.output_repo_dir, exist_ok=True)

    run_baseline_pipeline(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        paper_format="JSON",
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=None,
        stages=args.stages,
    )

    print(f"[DONE] stages: {args.stages}")
    print(f"[DONE] output_dir: {args.output_dir}")
    print(f"[DONE] output_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] planning trajectories: {os.path.join(args.output_dir, 'planning_trajectories.json')}")
    print(f"[DONE] api contract: {os.path.join(args.output_dir, 'api_predefine_contract.pyi')}")
    print(f"[DONE] coding artifacts: {os.path.join(args.output_dir, 'coding_artifacts')}")
    print(f"[DONE] generated repo entry: {os.path.join(args.output_repo_dir, 'main.py')}")
    print(f"[DONE] run_tag: {run_tag}")


if __name__ == "__main__":
    setup_logging()
    main()
