import argparse
import os
import re
import shutil
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval
from pipeline.feature_agent import DEFAULT_STAGES as FEATURE_DEFAULT_STAGES, run_feature_pipeline
from scripts.resume_feature_coding_round import (
    _cleanup_previous_outputs,
    _ensure_feature_rpg,
    _prepare_live_repo,
    _require_feature_analyzing_artifacts,
    _require_feature_planning_artifacts,
)
from workflow.feature_agent.rpg_coding import run_rpg_coding


def _infer_round_paths(path_hint: str) -> dict:
    normalized = os.path.abspath(path_hint).replace("\\", "/").rstrip("/")
    repo_pattern = re.compile(
        r"^(?P<prefix>.*?/outputs)/multi_paper_full_repro_rounds_repo_(?P<tag>\d{8}_\d{6})/"
        r"(?P<paper>[^/]+)/round_(?P<round>\d+)_repo$"
    )
    round_pattern = re.compile(
        r"^(?P<prefix>.*?/outputs)/multi_paper_full_repro_rounds_(?P<tag>\d{8}_\d{6})/"
        r"(?P<paper>[^/]+)/round_(?P<round>\d+)$"
    )
    match = repo_pattern.match(normalized)
    output_repo_dir = normalized
    if match:
        prefix = match.group("prefix")
        tag = match.group("tag")
        paper = match.group("paper")
        round_idx = match.group("round")
        round_root = os.path.join(
            prefix,
            f"multi_paper_full_repro_rounds_{tag}",
            paper,
            f"round_{round_idx}",
        )
    else:
        match = round_pattern.match(normalized)
        if not match:
            raise ValueError(
                "Could not infer multi-paper round paths. Expected either "
                ".../outputs/multi_paper_full_repro_rounds_<tag>/<paper>/round_<n> "
                "or .../outputs/multi_paper_full_repro_rounds_repo_<tag>/<paper>/round_<n>_repo"
            )
        prefix = match.group("prefix")
        tag = match.group("tag")
        paper = match.group("paper")
        round_idx = match.group("round")
        round_root = normalized
        output_repo_dir = os.path.join(
            prefix,
            f"multi_paper_full_repro_rounds_repo_{tag}",
            paper,
            f"round_{round_idx}_repo",
        )
    return {
        "paper_name": paper,
        "round_root": round_root,
        "feature_output_dir": os.path.join(round_root, "feature"),
        "baseline_repo_dir": os.path.join(round_root, "baseline_repo"),
        "baseline_interface_stub": os.path.join(round_root, "baseline", "interface_stubs_combined.py"),
        "eval_result_dir": os.path.join(round_root, "eval_results"),
        "output_repo_dir": output_repo_dir,
    }


def _resolve_pdf_json_paths(pdf_json_path: str) -> tuple[str, str]:
    direct_path = os.path.abspath(pdf_json_path)
    if direct_path.endswith("_cleaned.json"):
        planning_path = direct_path[: -len("_cleaned.json")] + ".json"
        if not os.path.isfile(planning_path):
            planning_path = direct_path
        return planning_path, direct_path

    cleaned_path = direct_path.replace(".json", "_cleaned.json")
    if os.path.isfile(cleaned_path):
        return direct_path, cleaned_path
    return direct_path, direct_path


def _has_feature_planning_artifacts(feature_output_dir: str) -> bool:
    config_path = os.path.join(feature_output_dir, "planning_config.yaml")
    response_path = os.path.join(feature_output_dir, "planning_response.json")
    traj_path = os.path.join(feature_output_dir, "planning_trajectories.json")
    return os.path.exists(config_path) and (os.path.exists(response_path) or os.path.exists(traj_path))


def _has_feature_analyzing_artifacts(feature_output_dir: str) -> bool:
    try:
        _require_feature_analyzing_artifacts(feature_output_dir)
        return True
    except FileNotFoundError:
        return False


def _cleanup_feature_dir(feature_output_dir: str) -> None:
    if os.path.exists(feature_output_dir):
        shutil.rmtree(feature_output_dir)
    os.makedirs(feature_output_dir, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume feature coding/eval for a multi-paper round from round root or round_<n>_repo."
    )
    parser.add_argument(
        "--round_root",
        type=str,
        default="",
        help="Optional multi-paper round root, e.g. .../multi_paper_full_repro_rounds_<tag>/<paper>/round_1",
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default=os.path.join(
            "outputs",
            "multi_paper_full_repro_rounds_repo_20260409_192648",
            "llm-detector-evasion",
            "round_1_repo",
        ),
        help="Repo directory of a multi-paper round, or leave default and use --round_root.",
    )
    parser.add_argument("--paper_name", type=str, default="")
    parser.add_argument("--feature_output_dir", type=str, default="")
    parser.add_argument("--baseline_repo_dir", type=str, default="")
    parser.add_argument("--baseline_interface_stub", type=str, default="")
    parser.add_argument("--eval_result_dir", type=str, default="")
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data",
            "paper2code",
            "paper2code_data",
            "iclr2024",
            "llm-detector-evasion.json",
        ),
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "llm-detector-evasion"),
    )
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument(
        "--feature_stages",
        type=str,
        nargs="+",
        default=list(FEATURE_DEFAULT_STAGES),
        help="Used only when feature artifacts are missing and the script restarts feature from planning.",
    )
    parser.add_argument("--skip_cleanup", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_coding", action="store_true")
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    round_locator = args.round_root or args.output_repo_dir
    inferred = _infer_round_paths(round_locator)
    args.output_repo_dir = os.path.abspath(inferred["output_repo_dir"])
    args.paper_name = args.paper_name or inferred["paper_name"]
    args.feature_output_dir = os.path.abspath(args.feature_output_dir or inferred["feature_output_dir"])
    args.baseline_repo_dir = os.path.abspath(args.baseline_repo_dir or inferred["baseline_repo_dir"])
    args.baseline_interface_stub = os.path.abspath(
        args.baseline_interface_stub or inferred["baseline_interface_stub"]
    )
    args.eval_result_dir = os.path.abspath(args.eval_result_dir or inferred["eval_result_dir"])
    feature_pdf_json_path, direct_pdf_json_path = _resolve_pdf_json_paths(args.pdf_json_path)
    args.pdf_json_path = os.path.abspath(args.pdf_json_path)
    args.gold_repo_dir = os.path.abspath(args.gold_repo_dir)

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    if os.path.exists(args.output_repo_dir) and not os.path.isdir(args.output_repo_dir):
        raise FileNotFoundError(f"output_repo_dir exists but is not a directory: {args.output_repo_dir}")
    if not os.path.isdir(args.baseline_repo_dir):
        raise FileNotFoundError(f"baseline_repo_dir not found: {args.baseline_repo_dir}")
    if not os.path.isfile(args.baseline_interface_stub):
        raise FileNotFoundError(f"interface_stubs_combined.py not found: {args.baseline_interface_stub}")
    if not os.path.isfile(feature_pdf_json_path):
        raise FileNotFoundError(f"feature planning pdf_json_path not found: {feature_pdf_json_path}")
    if not os.path.isfile(direct_pdf_json_path):
        raise FileNotFoundError(f"feature coding/eval pdf_json_path not found: {direct_pdf_json_path}")
    if not args.skip_eval_ref_based and not os.path.isdir(args.gold_repo_dir):
        raise FileNotFoundError(f"gold_repo_dir not found: {args.gold_repo_dir}")

    has_planning = _has_feature_planning_artifacts(args.feature_output_dir)
    has_analyzing = has_planning and _has_feature_analyzing_artifacts(args.feature_output_dir)
    resume_from_coding = has_planning and has_analyzing

    if not args.skip_cleanup:
        if resume_from_coding:
            _cleanup_previous_outputs(args.feature_output_dir, args.output_repo_dir, args.eval_result_dir)
        else:
            _cleanup_feature_dir(args.feature_output_dir)
            if os.path.exists(args.eval_result_dir):
                shutil.rmtree(args.eval_result_dir)
            if os.path.exists(args.output_repo_dir):
                shutil.rmtree(args.output_repo_dir)

    if not args.skip_feature:
        if resume_from_coding:
            _require_feature_planning_artifacts(args.feature_output_dir)
            _require_feature_analyzing_artifacts(args.feature_output_dir)
            _prepare_live_repo(args.feature_output_dir, args.output_repo_dir, args.baseline_repo_dir)
            _ensure_feature_rpg(args.feature_output_dir, args.baseline_interface_stub)

            if not args.skip_coding:
                run_rpg_coding(
                    paper_name=args.paper_name,
                    gpt_version=args.gpt_version,
                    output_dir=args.feature_output_dir,
                    output_repo_dir=args.output_repo_dir,
                    paper_format="JSON",
                    pdf_json_path=direct_pdf_json_path,
                    pdf_latex_path=None,
                    prompt_set="feature",
                    baseline_repo_dir=args.baseline_repo_dir,
                    live_repo_dir=args.output_repo_dir,
                    baseline_interface_stub_path=args.baseline_interface_stub,
                )
        else:
            run_feature_pipeline(
                paper_name=args.paper_name,
                gpt_version=args.gpt_version,
                output_dir=args.feature_output_dir,
                output_repo_dir=args.output_repo_dir,
                baseline_repo_dir=args.baseline_repo_dir,
                paper_format="JSON",
                pdf_json_path=feature_pdf_json_path,
                pdf_latex_path=None,
                stages=args.feature_stages,
                baseline_interface_stub_path=args.baseline_interface_stub,
            )

    os.makedirs(args.eval_result_dir, exist_ok=True)

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=args.feature_output_dir,
            pdf_json_path=direct_pdf_json_path,
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
            pdf_json_path=direct_pdf_json_path,
            target_repo_dir=args.output_repo_dir,
            eval_result_dir=args.eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] paper_name: {args.paper_name}")
    print(f"[DONE] feature_output_dir: {args.feature_output_dir}")
    print(f"[DONE] output_repo_dir: {args.output_repo_dir}")
    print(f"[DONE] baseline_repo_dir: {args.baseline_repo_dir}")
    print(f"[DONE] baseline_interface_stub: {args.baseline_interface_stub}")
    print(f"[DONE] eval_result_dir: {args.eval_result_dir}")
    print(f"[DONE] resume_from_coding: {resume_from_coding}")


if __name__ == "__main__":
    setup_logging()
    main()
