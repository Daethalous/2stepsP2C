import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import sys
from typing import Dict, List


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.logger import setup_logging
from evaluation.eval import run_eval
from pipeline.baseline_agent import run_baseline_pipeline
from pipeline.feature_agent import run_feature_pipeline
from scripts._run_path_utils import make_run_tag, resolve_default_output_path


PAPER_CONFIGS: Dict[str, Dict[str, str]] = {
    "llm-detector-evasion": {
        "pdf_json_path": os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "llm-detector-evasion.json"
        ),
        "gold_repo_dir": os.path.join("data", "paper2code", "gold_repos", "llm-detector-evasion"),
    },
    "TTM": {
        "pdf_json_path": os.path.join(
            "data", "paper2code", "paper2code_data", "iclr2024", "TTM.json"
        ),
        "gold_repo_dir": os.path.join("data", "paper2code", "gold_repos", "TTM"),
    },
    "SMM": {
        "pdf_json_path": os.path.join(
            "data", "paper2code", "paper2code_data", "icml2024", "SMM.json"
        ),
        "gold_repo_dir": os.path.join("data", "paper2code", "gold_repos", "SMM"),
    },
}


def _cleanup_dirs(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def _ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _find_eval_json(
    eval_result_dir: str,
    paper_name: str,
    eval_type: str,
    gpt_version: str,
) -> str:
    pattern = os.path.join(
        eval_result_dir, f"{paper_name}_eval_{eval_type}_{gpt_version}_*.json"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Eval result not found for pattern: {pattern}")
    return max(matches, key=os.path.getmtime)


def _load_eval_metrics(eval_json_path: str) -> Dict[str, object]:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    eval_result = payload["eval_result"]
    return {
        "score": float(eval_result["score"]),
        "valid_n": int(eval_result["valid_n"]),
        "score_list": json.dumps(eval_result.get("scroe_lst", []), ensure_ascii=False),
    }


def _write_detailed_csv(csv_path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "paper_name",
        "round_idx",
        "gpt_version",
        "output_dir",
        "output_repo_dir",
        "ref_free_score",
        "ref_free_valid_n",
        "ref_free_score_list",
        "ref_free_eval_json",
        "ref_based_score",
        "ref_based_valid_n",
        "ref_based_score_list",
        "ref_based_eval_json",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(csv_path: str, rows: List[Dict[str, object]]) -> None:
    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["paper_name"]), []).append(row)

    for paper_name, paper_rows in grouped.items():
        ref_free_scores = [float(r["ref_free_score"]) for r in paper_rows]
        ref_based_scores = [float(r["ref_based_score"]) for r in paper_rows]
        summary_rows.append(
            {
                "paper_name": paper_name,
                "rounds": len(paper_rows),
                "ref_free_mean": statistics.mean(ref_free_scores),
                "ref_free_min": min(ref_free_scores),
                "ref_free_max": max(ref_free_scores),
                "ref_based_mean": statistics.mean(ref_based_scores),
                "ref_based_min": min(ref_based_scores),
                "ref_based_max": max(ref_based_scores),
            }
        )

    fieldnames = [
        "paper_name",
        "rounds",
        "ref_free_mean",
        "ref_free_min",
        "ref_free_max",
        "ref_based_mean",
        "ref_based_min",
        "ref_based_max",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _run_single_round(
    paper_name: str,
    paper_cfg: Dict[str, str],
    round_idx: int,
    gpt_version: str,
    output_root_dir: str,
    output_repo_root_dir: str,
    generated_n: int,
    baseline_stages: List[str],
    feature_stages: List[str],
    skip_cleanup: bool,
) -> Dict[str, object]:
    output_dir = os.path.join(output_root_dir, paper_name, f"round_{round_idx}")
    output_repo_dir = os.path.join(output_repo_root_dir, paper_name, f"round_{round_idx}_repo")

    baseline_output_dir = os.path.join(output_dir, "baseline")
    baseline_snapshot_dir = os.path.join(output_dir, "baseline_repo")
    feature_output_dir = os.path.join(output_dir, "feature")
    eval_result_dir = os.path.join(output_dir, "eval_results")

    if not skip_cleanup:
        _cleanup_dirs(
            baseline_output_dir,
            baseline_snapshot_dir,
            feature_output_dir,
            eval_result_dir,
            output_repo_dir,
        )
    else:
        for path in (
            baseline_output_dir,
            baseline_snapshot_dir,
            feature_output_dir,
            eval_result_dir,
            output_repo_dir,
        ):
            os.makedirs(path, exist_ok=True)

    pdf_json_path = paper_cfg["pdf_json_path"]
    cleaned_json_path = pdf_json_path.replace(".json", "_cleaned.json")
    gold_repo_dir = paper_cfg["gold_repo_dir"]

    _ensure_exists(pdf_json_path, "pdf_json_path")
    _ensure_exists(cleaned_json_path, "cleaned_json_path")
    _ensure_exists(gold_repo_dir, "gold_repo_dir")

    run_baseline_pipeline(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=baseline_output_dir,
        output_repo_dir=output_repo_dir,
        paper_format="JSON",
        pdf_json_path=pdf_json_path,
        pdf_latex_path=None,
        stages=baseline_stages,
    )

    if os.path.exists(baseline_snapshot_dir):
        shutil.rmtree(baseline_snapshot_dir)
    shutil.copytree(output_repo_dir, baseline_snapshot_dir)

    run_feature_pipeline(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=feature_output_dir,
        output_repo_dir=output_repo_dir,
        baseline_repo_dir=baseline_snapshot_dir,
        paper_format="JSON",
        pdf_json_path=pdf_json_path,
        pdf_latex_path=None,
        stages=feature_stages,
        baseline_interface_stub_path=os.path.join(
            baseline_output_dir, "interface_stubs_combined.py"
        ),
    )

    run_eval(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=feature_output_dir,
        pdf_json_path=cleaned_json_path,
        target_repo_dir=output_repo_dir,
        eval_result_dir=eval_result_dir,
        eval_type="ref_free",
        generated_n=generated_n,
        is_papercoder=True,
    )
    run_eval(
        paper_name=paper_name,
        gpt_version=gpt_version,
        output_dir=feature_output_dir,
        pdf_json_path=cleaned_json_path,
        target_repo_dir=output_repo_dir,
        eval_result_dir=eval_result_dir,
        eval_type="ref_based",
        gold_repo_dir=gold_repo_dir,
        generated_n=generated_n,
        is_papercoder=True,
    )

    ref_free_json = _find_eval_json(eval_result_dir, paper_name, "ref_free", gpt_version)
    ref_based_json = _find_eval_json(eval_result_dir, paper_name, "ref_based", gpt_version)
    ref_free_metrics = _load_eval_metrics(ref_free_json)
    ref_based_metrics = _load_eval_metrics(ref_based_json)

    return {
        "paper_name": paper_name,
        "round_idx": round_idx,
        "gpt_version": gpt_version,
        "output_dir": output_dir,
        "output_repo_dir": output_repo_dir,
        "ref_free_score": ref_free_metrics["score"],
        "ref_free_valid_n": ref_free_metrics["valid_n"],
        "ref_free_score_list": ref_free_metrics["score_list"],
        "ref_free_eval_json": ref_free_json,
        "ref_based_score": ref_based_metrics["score"],
        "ref_based_valid_n": ref_based_metrics["valid_n"],
        "ref_based_score_list": ref_based_metrics["score_list"],
        "ref_based_eval_json": ref_based_json,
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        type=str,
        nargs="+",
        default=["llm-detector-evasion", "TTM", "SMM"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--generated_n", type=int, default=8)
    default_output_root_dir = os.path.join("outputs", "multi_paper_full_repro_rounds")
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=default_output_root_dir,
    )
    default_output_repo_root_dir = os.path.join("outputs", "multi_paper_full_repro_rounds_repo")
    parser.add_argument(
        "--output_repo_root_dir",
        type=str,
        default=default_output_repo_root_dir,
    )
    default_detailed_csv_path = os.path.join(default_output_root_dir, "all_round_scores.csv")
    parser.add_argument(
        "--detailed_csv_path",
        type=str,
        default=default_detailed_csv_path,
    )
    default_summary_csv_path = os.path.join(default_output_root_dir, "paper_score_summary.csv")
    parser.add_argument(
        "--summary_csv_path",
        type=str,
        default=default_summary_csv_path,
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix for output paths. Defaults to a current timestamp when using default output paths.",
    )
    parser.add_argument(
        "--baseline_stages",
        type=str,
        nargs="+",
        default=["preprocess", "planning", "extract", "build_rpg", "analyzing", "interface_design", "coding", "typecheck"],
    )
    parser.add_argument(
        "--feature_stages",
        type=str,
        nargs="+",
        default=["planning", "extract", "build_feature_rpg", "analyzing", "coding"],
    )
    parser.add_argument("--skip_cleanup", action="store_true")
    args = parser.parse_args()

    unknown_papers = [paper for paper in args.papers if paper not in PAPER_CONFIGS]
    if unknown_papers:
        raise ValueError(f"Unsupported papers: {unknown_papers}")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    run_tag = args.run_tag or make_run_tag()
    args.output_root_dir = resolve_default_output_path(
        args.output_root_dir,
        default_output_root_dir,
        run_tag,
    )
    args.output_repo_root_dir = resolve_default_output_path(
        args.output_repo_root_dir,
        default_output_repo_root_dir,
        run_tag,
    )
    if os.path.normpath(args.detailed_csv_path) == os.path.normpath(default_detailed_csv_path):
        args.detailed_csv_path = os.path.join(args.output_root_dir, "all_round_scores.csv")
    if os.path.normpath(args.summary_csv_path) == os.path.normpath(default_summary_csv_path):
        args.summary_csv_path = os.path.join(args.output_root_dir, "paper_score_summary.csv")

    all_rows: List[Dict[str, object]] = []
    for paper_name in args.papers:
        paper_cfg = PAPER_CONFIGS[paper_name]
        for round_idx in range(1, args.rounds + 1):
            row = _run_single_round(
                paper_name=paper_name,
                paper_cfg=paper_cfg,
                round_idx=round_idx,
                gpt_version=args.gpt_version,
                output_root_dir=args.output_root_dir,
                output_repo_root_dir=args.output_repo_root_dir,
                generated_n=args.generated_n,
                baseline_stages=args.baseline_stages,
                feature_stages=args.feature_stages,
                skip_cleanup=args.skip_cleanup,
            )
            all_rows.append(row)
            _write_detailed_csv(args.detailed_csv_path, all_rows)
            _write_summary_csv(args.summary_csv_path, all_rows)

    print(f"[DONE] detailed_csv_path: {args.detailed_csv_path}")
    print(f"[DONE] summary_csv_path: {args.summary_csv_path}")
    print(f"[DONE] run_tag: {run_tag}")


if __name__ == "__main__":
    setup_logging()
    main()
