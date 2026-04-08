import argparse
import hashlib
import os
import re
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.data_loader import load_pipeline_context, sanitize_todo_file_name
from core.logger import setup_logging
from core.utils import merge_yaml_configs
from evaluation.eval import run_eval
from workflow.coding import run_coding


def _make_safe_artifact_stem(path_text: str) -> str:
    normalized = sanitize_todo_file_name(path_text) or str(path_text or "").strip()
    normalized = normalized.replace("\\", "/")
    readable = normalized.replace("/", "__")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("._")
    readable = re.sub(r"_+", "_", readable)
    if not readable:
        readable = "artifact"
    suffix = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def _find_analysis_response_path(feature_output_dir: str, todo_file_name: str) -> str:
    candidate = sanitize_todo_file_name(todo_file_name) or todo_file_name
    safe_name = _make_safe_artifact_stem(candidate)
    hashed_path = os.path.join(feature_output_dir, f"{safe_name}_simple_analysis_response.json")
    legacy_path = os.path.join(feature_output_dir, f"{candidate.replace('/', '_')}_simple_analysis_response.json")
    if os.path.exists(hashed_path):
        return hashed_path
    if os.path.exists(legacy_path):
        return legacy_path
    return ""


def _require_feature_planning_artifacts(feature_output_dir: str) -> None:
    response_path = os.path.join(feature_output_dir, "planning_response.json")
    traj_path = os.path.join(feature_output_dir, "planning_trajectories.json")
    config_path = os.path.join(feature_output_dir, "planning_config.yaml")
    missing = [path for path in (config_path,) if not os.path.exists(path)]
    if not os.path.exists(response_path) and not os.path.exists(traj_path):
        missing.extend([response_path, traj_path])
    if missing:
        raise FileNotFoundError(
            "Missing feature planning artifacts under feature_output_dir.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _require_feature_analyzing_artifacts(feature_output_dir: str) -> None:
    ctx = load_pipeline_context(feature_output_dir)
    missing = []
    for todo_file_name in ctx.todo_file_lst:
        if todo_file_name == "config.yaml":
            continue
        response_path = _find_analysis_response_path(feature_output_dir, todo_file_name)
        if not response_path:
            missing.append(todo_file_name)
    if missing:
        raise FileNotFoundError(
            "Missing feature analyzing artifacts under feature_output_dir.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _cleanup_previous_outputs(feature_output_dir: str, output_repo_dir: str, eval_result_dir: str) -> None:
    coding_artifacts_dir = os.path.join(feature_output_dir, "coding_artifacts")
    for path in (coding_artifacts_dir, eval_result_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists(output_repo_dir):
        shutil.rmtree(output_repo_dir)


def _prepare_live_repo(feature_output_dir: str, output_repo_dir: str, baseline_repo_dir: str) -> None:
    if not os.path.isdir(output_repo_dir):
        shutil.copytree(baseline_repo_dir, output_repo_dir)

    base_cfg = os.path.join(output_repo_dir, "config.yaml")
    overlay_cfg = os.path.join(feature_output_dir, "planning_config.yaml")
    if os.path.exists(base_cfg) and os.path.exists(overlay_cfg):
        merge_yaml_configs(
            base_path=base_cfg,
            overlay_path=overlay_cfg,
            output_path=base_cfg,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str, default="llm-detector-evasion")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--feature_output_dir",
        type=str,
        default=os.path.join(
            "outputs",
            "multi_paper_full_repro_rounds_20260406_151229",
            "llm-detector-evasion",
            "round_1",
            "feature",
        ),
        help="Feature experiment dir containing planning/extract/analyzing artifacts",
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join(
            "outputs",
            "multi_paper_full_repro_rounds_repo_20260406_151229",
            "llm-detector-evasion",
            "round_1_repo",
        ),
        help="Live repo resumed from the previous round output",
    )
    parser.add_argument(
        "--baseline_repo_dir",
        type=str,
        default=os.path.join(
            "outputs",
            "multi_paper_full_repro_rounds_20260406_151229",
            "llm-detector-evasion",
            "round_1",
            "baseline_repo",
        ),
        help="Baseline repo snapshot used as the feature injection base",
    )
    parser.add_argument(
        "--api_predefine_contract",
        type=str,
        default=os.path.join(
            "outputs",
            "multi_paper_full_repro_rounds_20260406_151229",
            "llm-detector-evasion",
            "round_1",
            "baseline",
            "api_predefine_contract.pyi",
        ),
        help="Path to baseline api_predefine_contract.pyi for feature coding prompt",
    )
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data",
            "paper2code",
            "paper2code_data",
            "iclr2024",
            "llm-detector-evasion_cleaned.json",
        ),
        help="Cleaned paper JSON used by feature coding/evaluation",
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "llm-detector-evasion"),
        help="Gold repo used for ref-based evaluation",
    )
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        default="",
        help="Defaults to <feature_output_dir>/eval_results when omitted",
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

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    if not args.eval_result_dir:
        args.eval_result_dir = os.path.join(args.feature_output_dir, "eval_results")

    _require_feature_planning_artifacts(args.feature_output_dir)
    _require_feature_analyzing_artifacts(args.feature_output_dir)

    if not os.path.isdir(args.baseline_repo_dir):
        raise FileNotFoundError(f"baseline_repo_dir not found: {args.baseline_repo_dir}")
    if not os.path.isfile(args.pdf_json_path):
        raise FileNotFoundError(f"pdf_json_path not found: {args.pdf_json_path}")
    if not os.path.isfile(args.api_predefine_contract):
        raise FileNotFoundError(f"api_predefine_contract.pyi not found: {args.api_predefine_contract}")
    if not args.skip_eval_ref_based and not os.path.isdir(args.gold_repo_dir):
        raise FileNotFoundError(f"gold_repo_dir not found: {args.gold_repo_dir}")

    if not args.skip_cleanup:
        _cleanup_previous_outputs(args.feature_output_dir, args.output_repo_dir, args.eval_result_dir)

    _prepare_live_repo(args.feature_output_dir, args.output_repo_dir, args.baseline_repo_dir)

    if not args.skip_coding:
        run_coding(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.feature_output_dir,
            output_repo_dir=args.output_repo_dir,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            prompt_set="feature",
            baseline_repo_dir=args.baseline_repo_dir,
            live_repo_dir=args.output_repo_dir,
            api_predefine_contract_path=args.api_predefine_contract,
        )

    os.makedirs(args.eval_result_dir, exist_ok=True)

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.feature_output_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    if not args.skip_eval_ref_based:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.feature_output_dir,
            pdf_json_path=args.pdf_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] feature_output_dir: {args.feature_output_dir}")
    print(f"[DONE] output_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] eval_result_dir: {args.eval_result_dir}")
    print(f"[DONE] api_predefine_contract: {args.api_predefine_contract}")


if __name__ == "__main__":
    setup_logging()
    main()
