"""
Run feature analyzing for iTransformer using planning artifacts from a prior feature planning+extract run.

Requires:
  - outputs/iTransformer_feature_planning_extract/planning_trajectories.json
  - outputs/iTransformer_feature_planning_extract/planning_config.yaml
  - A cleaned paper JSON path used by the feature planning run
  - baseline repo and live repo paths for feature-side file context
"""

import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from workflow.analyzing import run_analyzing


def _require_planning_artifacts(output_dir: str) -> None:
    traj = os.path.join(output_dir, "planning_trajectories.json")
    cfg = os.path.join(output_dir, "planning_config.yaml")
    missing = [p for p in (traj, cfg) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing feature planning outputs under output_dir. Run feature planning+extract first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, default="iTransformer")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_planning_extract"),
        help="Directory containing planning_trajectories.json and planning_config.yaml",
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "iTransformer_cleaned.json"
        ),
        help="Cleaned paper JSON used by feature planning",
    )
    parser.add_argument(
        "--baseline_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round_repo copy"),
        help="Baseline repo snapshot used for feature injection",
    )
    parser.add_argument(
        "--live_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_planning_extract_repo"),
        help="Live repo dir produced by feature planning+extract run",
    )
    parser.add_argument(
        "--api_predefine_contract",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round", "api_predefine_contract.pyi"),
        help="Path to api_predefine_contract.pyi consumed by feature analyzing prompt",
    )
    args = parser.parse_args()

    _require_planning_artifacts(args.output_dir)

    if not os.path.isfile(args.pdf_json_path):
        raise FileNotFoundError(f"cleaned pdf_json_path not found: {args.pdf_json_path}")
    if not os.path.isdir(args.baseline_repo_dir):
        raise FileNotFoundError(f"baseline_repo_dir not found: {args.baseline_repo_dir}")
    if not os.path.isdir(args.live_repo_dir):
        raise FileNotFoundError(f"live_repo_dir not found: {args.live_repo_dir}")
    if not os.path.isfile(args.api_predefine_contract):
        raise FileNotFoundError(
            f"api_predefine_contract.pyi not found: {args.api_predefine_contract}"
        )

    run_analyzing(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format="JSON",
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=None,
        prompt_set="feature",
        baseline_repo_dir=args.baseline_repo_dir,
        live_repo_dir=args.live_repo_dir,
        api_predefine_contract_path=args.api_predefine_contract,
    )

    print(f"[DONE] analyzing artifacts: {os.path.join(args.output_dir, 'analyzing_artifacts')}")
    print(f"[DONE] output_dir: {args.output_dir}")
    print(f"[DONE] baseline_repo_dir: {args.baseline_repo_dir}")
    print(f"[DONE] live_repo_dir: {args.live_repo_dir}")
    print(f"[DONE] api_predefine_contract: {args.api_predefine_contract}")
    print(f"[DONE] accumulated_cost: {os.path.join(args.output_dir, 'accumulated_cost.json')}")


if __name__ == "__main__":
    setup_logging()
    main()
