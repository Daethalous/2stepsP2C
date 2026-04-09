import argparse
import os
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.data_loader import load_pipeline_context
from core.logger import setup_logging
from core.utils import merge_yaml_configs
from evaluation.eval import run_eval
from workflow.coding import run_coding


def _require_feature_planning_artifacts(output_dir: str) -> None:
    response_path = os.path.join(output_dir, "planning_response.json")
    traj_path = os.path.join(output_dir, "planning_trajectories.json")
    config_path = os.path.join(output_dir, "planning_config.yaml")
    missing = [
        path
        for path in (config_path,)
        if not os.path.exists(path)
    ]
    if not os.path.exists(response_path) and not os.path.exists(traj_path):
        missing.extend([response_path, traj_path])
    if missing:
        raise FileNotFoundError(
            "Missing feature planning artifacts under output_dir. Run feature planning+extract first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _require_feature_analyzing_artifacts(output_dir: str) -> None:
    ctx = load_pipeline_context(output_dir)
    missing = []
    for todo_file_name in ctx.todo_file_lst:
        if todo_file_name == "config.yaml":
            continue
        save_name = todo_file_name.replace("/", "_")
        response_path = os.path.join(output_dir, f"{save_name}_simple_analysis_response.json")
        if not os.path.exists(response_path):
            missing.append(response_path)
    if missing:
        raise FileNotFoundError(
            "Missing feature analyzing artifacts under output_dir. Run feature analyzing first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _cleanup_previous_outputs(output_dir: str, output_repo_dir: str) -> None:
    coding_artifacts_dir = os.path.join(output_dir, "coding_artifacts")
    eval_result_dir = os.path.join(output_dir, "eval_results")
    for path in (coding_artifacts_dir, eval_result_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists(output_repo_dir):
        shutil.rmtree(output_repo_dir)


def _prepare_live_repo(output_dir: str, output_repo_dir: str, baseline_repo_dir: str) -> None:
    if not os.path.isdir(output_repo_dir):
        shutil.copytree(baseline_repo_dir, output_repo_dir)

    base_cfg = os.path.join(output_repo_dir, "config.yaml")
    overlay_cfg = os.path.join(output_dir, "planning_config.yaml")
    if os.path.exists(base_cfg) and os.path.exists(overlay_cfg):
        merge_yaml_configs(
            base_path=base_cfg,
            overlay_path=overlay_cfg,
            output_path=base_cfg,
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
        help="Feature experiment dir containing planning/analyzing artifacts",
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_feature_planning_extract_repo"),
        help="Live repo to continue from the previous feature planning/extract experiment",
    )
    parser.add_argument(
        "--baseline_repo_dir",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round_repo copy"),
        help="Baseline repo snapshot used as the feature injection base",
    )
    parser.add_argument(
        "--api_predefine_contract",
        type=str,
        default=os.path.join("outputs", "iTransformer_baseline_full_round", "api_predefine_contract.pyi"),
        help="Path to baseline api_predefine_contract.pyi for feature coding prompt",
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "iTransformer"),
        help="Gold repo used for ref-based evaluation",
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "iTransformer_cleaned.json"
        ),
        help="Cleaned paper JSON used by feature analyzing/coding/evaluation",
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument(
        "--skip_cleanup",
        action="store_true",
        help="Keep existing coding_artifacts, eval_results, and output_repo_dir",
    )
    parser.add_argument("--skip_coding", action="store_true")
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    _require_feature_planning_artifacts(args.output_dir)
    _require_feature_analyzing_artifacts(args.output_dir)

    if not os.path.isdir(args.baseline_repo_dir):
        raise FileNotFoundError(f"baseline_repo_dir not found: {args.baseline_repo_dir}")
    if not os.path.isfile(args.pdf_json_path):
        raise FileNotFoundError(f"cleaned pdf_json_path not found: {args.pdf_json_path}")
    if not os.path.isfile(args.api_predefine_contract):
        raise FileNotFoundError(
            f"api_predefine_contract.pyi not found: {args.api_predefine_contract}"
        )
    if not args.skip_eval_ref_based and not os.path.isdir(args.gold_repo_dir):
        raise FileNotFoundError(f"gold_repo_dir not found: {args.gold_repo_dir}")

    if not args.skip_cleanup:
        _cleanup_previous_outputs(args.output_dir, args.output_repo_dir)

    _prepare_live_repo(args.output_dir, args.output_repo_dir, args.baseline_repo_dir)

    if not args.skip_coding:
        run_coding(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            prompt_set="feature",
            baseline_repo_dir=args.baseline_repo_dir,
            live_repo_dir=args.output_repo_dir,
            api_predefine_contract_path=args.api_predefine_contract,
        )

    eval_result_dir = os.path.join(args.output_dir, "eval_results")

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.output_dir,
            pdf_json_path=args.pdf_json_path,
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
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] output_dir: {args.output_dir}")
    print(f"[DONE] output_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] coding_artifacts: {os.path.join(args.output_dir, 'coding_artifacts')}")
    print(f"[DONE] eval_result_dir: {eval_result_dir}")
    print(f"[DONE] api_predefine_contract: {args.api_predefine_contract}")


if __name__ == "__main__":
    setup_logging()
    main()
