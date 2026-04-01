import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.data_loader import load_pipeline_context
from core.logger import setup_logging
from workflow.api_predefine import run_api_predefine


def _require_planning_artifacts(output_dir: str) -> None:
    traj = os.path.join(output_dir, "planning_trajectories.json")
    cfg = os.path.join(output_dir, "planning_config.yaml")
    missing = [p for p in (traj, cfg) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing planning outputs under output_dir. Run planning+extract first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _require_analyzing_artifacts(output_dir: str) -> None:
    ctx = load_pipeline_context(output_dir)
    missing = []
    for todo_file_name in ctx.todo_file_lst:
        if todo_file_name == "config.yaml":
            continue
        save_todo_file_name = todo_file_name.replace("/", "_")
        response_path = os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_response.json")
        if not os.path.exists(response_path):
            missing.append(response_path)
    if missing:
        raise FileNotFoundError(
            "Missing analyzing outputs under output_dir. Run baseline analyzing first.\n"
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
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "iTransformer_cleaned.json"
        ),
    )
    args = parser.parse_args()

    _require_planning_artifacts(args.output_dir)
    _require_analyzing_artifacts(args.output_dir)

    run_api_predefine(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format="JSON",
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=None,
        prompt_set="baseline",
    )

    print(f"[DONE] api contract: {os.path.join(args.output_dir, 'api_predefine_contract.pyi')}")
    print(f"[DONE] api artifacts: {os.path.join(args.output_dir, 'api_predefine_artifacts')}")
    print(f"[DONE] accumulated_cost: {os.path.join(args.output_dir, 'accumulated_cost.json')}")


if __name__ == "__main__":
    setup_logging()
    main()
