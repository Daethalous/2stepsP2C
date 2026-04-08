import argparse
import os
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from core.parser.pdf_process import run_pdf_process
from pipeline.feature_agent import run_feature_pipeline


def run_experiment(
    paper_name: str,
    gpt_version: str,
    pdf_json_path: str,
    output_dir: str,
    output_repo_dir: str,
    baseline_repo_dir: str,
    api_predefine_contract_path: str,
    force_preprocess: bool = False,
    reset_output_repo: bool = False,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")
    if not os.path.isdir(baseline_repo_dir):
        raise FileNotFoundError(f"baseline_repo_dir not found: {baseline_repo_dir}")
    if not os.path.isfile(pdf_json_path):
        raise FileNotFoundError(f"pdf_json_path not found: {pdf_json_path}")
    if not os.path.isfile(api_predefine_contract_path):
        raise FileNotFoundError(
            f"api_predefine_contract.pyi not found: {api_predefine_contract_path}"
        )

    os.makedirs(output_dir, exist_ok=True)

    cleaned_json_path = pdf_json_path.replace(".json", "_cleaned.json")
    if force_preprocess or not os.path.exists(cleaned_json_path):
        run_pdf_process(pdf_json_path, cleaned_json_path)

    if reset_output_repo and os.path.isdir(output_repo_dir):
        shutil.rmtree(output_repo_dir)

    run_feature_pipeline(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        baseline_repo_dir=baseline_repo_dir,
        paper_format="JSON",
        pdf_json_path=pdf_json_path,
        pdf_latex_path=None,
        stages=["planning", "extract"],
        api_predefine_contract_path=api_predefine_contract_path,
    )

    print(f"[DONE] cleaned json: {cleaned_json_path}")
    print(f"[DONE] baseline_repo_dir: {baseline_repo_dir}")
    print(f"[DONE] api_predefine_contract: {api_predefine_contract_path}")
    print(f"[DONE] output_dir: {output_dir}")
    print(f"[DONE] output_repo_dir: {output_repo_dir}")
    print(f"[DONE] planning trajectories: {os.path.join(output_dir, 'planning_trajectories.json')}")
    print(f"[DONE] planning config: {os.path.join(output_dir, 'planning_config.yaml')}")
    print(f"[DONE] planning artifacts: {os.path.join(output_dir, 'planning_artifacts')}")
    print(f"[DONE] merged repo config: {os.path.join(output_repo_dir, 'config.yaml')}")


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
        "--baseline_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round_repo copy"),
    )
    parser.add_argument(
        "--api_predefine_contract",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round", "api_predefine_contract.pyi"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_planning_extract"),
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_planning_extract_repo"),
    )
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--reset_output_repo", action="store_true")
    args = parser.parse_args()

    run_experiment(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        pdf_json_path=args.pdf_json_path,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        baseline_repo_dir=args.baseline_repo_dir,
        api_predefine_contract_path=args.api_predefine_contract,
        force_preprocess=args.force_preprocess,
        reset_output_repo=args.reset_output_repo,
    )
