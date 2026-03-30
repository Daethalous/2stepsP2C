import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from pipeline.baseline_agent import run_baseline_pipeline


def run_experiment(
    paper_name: str,
    gpt_version: str,
    pdf_json_path: str,
    output_dir: str,
    output_repo_dir: str,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_repo_dir, exist_ok=True)

    # Complete baseline planning + extract_artifacts path:
    # preprocess -> planning -> extract
    run_baseline_pipeline(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        paper_format="JSON",
        pdf_json_path=pdf_json_path,
        pdf_latex_path=None,
        stages=["preprocess", "planning", "extract"],
    )

    print(f"[DONE] output_dir: {output_dir}")
    print(f"[DONE] output_repo_dir: {output_repo_dir}")
    print(f"[DONE] planning trajectories: {os.path.join(output_dir, 'planning_trajectories.json')}")
    print(f"[DONE] planning config: {os.path.join(output_dir, 'planning_config.yaml')}")
    print(f"[DONE] extracted config copy: {os.path.join(output_repo_dir, 'config.yaml')}")


if __name__ == "__main__":
    setup_logging()

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
        default=os.path.join("outputs", "iTransformer_baseline_planning_extract"),
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_planning_extract_repo"),
    )
    args = parser.parse_args()

    run_experiment(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        pdf_json_path=args.pdf_json_path,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
    )
