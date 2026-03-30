"""
Run baseline analyzing for iTransformer using planning artifacts from a prior run.

Requires:
  - outputs/iTransformer_baseline_planning_extract/planning_trajectories.json
  - outputs/iTransformer_baseline_planning_extract/planning_config.yaml
  - Same iTransformer_cleaned.json path used when planning was run (default: next to source JSON).
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
            "Missing planning outputs under output_dir. Run planning+extract first.\n"
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
        default=os.path.join("outputs", "iTransformer_baseline_planning_extract"),
        help="Directory containing planning_trajectories.json and planning_config.yaml",
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "iTransformer_cleaned.json"
        ),
        help="Cleaned paper JSON (same as used for baseline planning)",
    )
    args = parser.parse_args()

    _require_planning_artifacts(args.output_dir)

    run_analyzing(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format="JSON",
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=None,
        prompt_set="baseline",
        baseline_repo_dir=None,
        live_repo_dir=None,
    )

    print(f"[DONE] analyzing artifacts: {os.path.join(args.output_dir, 'analyzing_artifacts')}")
    print(f"[DONE] accumulated_cost: {os.path.join(args.output_dir, 'accumulated_cost.json')}")


if __name__ == "__main__":
    setup_logging()
    main()
