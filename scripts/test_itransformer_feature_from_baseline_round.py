"""
在既有 full repro round 的 baseline 产物上，单独跑新版（RPG 驱动）feature 流水线。

默认以 outputs/iTransformer_full_repro_round_20260409_144053 为 round 根目录：
  - baseline 产物：{round_root}/baseline（含 interface_stubs_combined.py、rpg_graph.json 等）
  - 代码快照：优先 {round_root}/baseline_repo；若不存在则尝试同时间戳的
    outputs/iTransformer_full_repro_round_repo_<tag>/

feature 输出与 live repo 默认写到带时间戳的新目录，避免覆盖原 round 的 feature/repo。
"""
from __future__ import annotations

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
from scripts._run_path_utils import make_run_tag


def _infer_repo_dir_from_round_root(round_root: str) -> str:
    """若 round 根目录下无 baseline_repo，则根据目录名中的时间戳推断并行 repo 路径。"""
    base = os.path.basename(os.path.normpath(round_root))
    m = re.search(r"(\d{8}_\d{6})$", base)
    if not m:
        return ""
    tag = m.group(1)
    parent = os.path.dirname(os.path.abspath(round_root))
    candidate = os.path.join(parent, f"iTransformer_full_repro_round_repo_{tag}")
    return candidate if os.path.isdir(candidate) else ""


def _resolve_baseline_repo(round_root: str, explicit: str) -> str:
    if explicit:
        return os.path.abspath(explicit)
    nested = os.path.join(round_root, "baseline_repo")
    if os.path.isdir(nested):
        return os.path.abspath(nested)
    inferred = _infer_repo_dir_from_round_root(round_root)
    if inferred:
        return os.path.abspath(inferred)
    return ""


def _cleanup_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在指定 iTransformer full repro round 的 baseline 上测试新版 feature（含 build_feature_rpg）。"
    )
    parser.add_argument(
        "--round_root",
        type=str,
        default=os.path.join(
            "outputs",
            "iTransformer_full_repro_round_20260409_144053",
        ),
        help="该次实验的根目录（其下应有 baseline/）",
    )
    parser.add_argument(
        "--baseline_artifacts_dir",
        type=str,
        default="",
        help="baseline 阶段输出目录（默认 {round_root}/baseline）",
    )
    parser.add_argument(
        "--baseline_repo_dir",
        type=str,
        default="",
        help="baseline 代码快照目录（默认 {round_root}/baseline_repo 或推断的 *_repo_<tag>）",
    )
    parser.add_argument(
        "--feature_output_dir",
        type=str,
        default="",
        help="新版 feature 产物目录；默认 {round_root}/feature_rpg_<timestamp>",
    )
    parser.add_argument(
        "--output_repo_dir",
        type=str,
        default="",
        help="本次 feature 写入的 live repo；默认与 round 同级的 iTransformer_feature_rpg_repo_<timestamp>",
    )
    parser.add_argument("--paper_name", type=str, default="iTransformer")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--pdf_json_path",
        type=str,
        default=os.path.join(
            "data",
            "paper2code",
            "paper2code_data",
            "iclr2024",
            "iTransformer.json",
        ),
    )
    parser.add_argument(
        "--gold_repo_dir",
        type=str,
        default=os.path.join("data", "paper2code", "gold_repos", "iTransformer"),
    )
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument(
        "--feature_stages",
        type=str,
        nargs="+",
        default=list(FEATURE_DEFAULT_STAGES),
    )
    parser.add_argument("--skip_cleanup", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_eval_ref_free", action="store_true")
    parser.add_argument("--skip_eval_ref_based", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required to run this script.")

    round_root = os.path.abspath(args.round_root)
    if not os.path.isdir(round_root):
        raise FileNotFoundError(f"round_root not found: {round_root}")

    baseline_artifacts = os.path.abspath(
        args.baseline_artifacts_dir or os.path.join(round_root, "baseline")
    )
    stub_path = os.path.join(baseline_artifacts, "interface_stubs_combined.py")
    if not os.path.isfile(stub_path):
        raise FileNotFoundError(
            f"Missing baseline interface stub (expected RPG baseline artifact): {stub_path}"
        )

    baseline_repo = _resolve_baseline_repo(round_root, args.baseline_repo_dir)
    if not baseline_repo or not os.path.isdir(baseline_repo):
        raise FileNotFoundError(
            "Could not resolve baseline_repo_dir. "
            f"Pass --baseline_repo_dir explicitly. Tried under {round_root} and inferred sibling repo."
        )

    run_tag = make_run_tag()
    parent = os.path.dirname(round_root)

    if args.feature_output_dir:
        feature_output_dir = os.path.abspath(args.feature_output_dir)
    else:
        feature_output_dir = os.path.join(round_root, f"feature_rpg_{run_tag}")

    if args.output_repo_dir:
        output_repo_dir = os.path.abspath(args.output_repo_dir)
    else:
        output_repo_dir = os.path.join(
            parent,
            f"iTransformer_feature_rpg_repo_{run_tag}",
        )

    if not args.skip_cleanup:
        _cleanup_dir(feature_output_dir)
        _cleanup_dir(output_repo_dir)
    else:
        os.makedirs(feature_output_dir, exist_ok=True)
        os.makedirs(output_repo_dir, exist_ok=True)

    cleaned_json_path = args.pdf_json_path.replace(".json", "_cleaned.json")
    if not os.path.isfile(args.pdf_json_path):
        raise FileNotFoundError(f"pdf_json_path not found: {args.pdf_json_path}")
    if not args.skip_eval_ref_based and not os.path.isdir(args.gold_repo_dir):
        raise FileNotFoundError(f"gold_repo_dir not found: {args.gold_repo_dir}")

    if not args.skip_feature:
        shutil.copytree(baseline_repo, output_repo_dir, dirs_exist_ok=True)
        run_feature_pipeline(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            output_repo_dir=output_repo_dir,
            baseline_repo_dir=baseline_repo,
            paper_format="JSON",
            pdf_json_path=args.pdf_json_path,
            pdf_latex_path=None,
            stages=args.feature_stages,
            baseline_interface_stub_path=stub_path,
        )

    eval_result_dir = os.path.join(feature_output_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)

    if not args.skip_eval_ref_free:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=output_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_free",
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    if not args.skip_eval_ref_based:
        run_eval(
            paper_name=args.paper_name,
            gpt_version=args.gpt_version,
            output_dir=feature_output_dir,
            pdf_json_path=cleaned_json_path,
            target_repo_dir=output_repo_dir,
            eval_result_dir=eval_result_dir,
            eval_type="ref_based",
            gold_repo_dir=args.gold_repo_dir,
            generated_n=args.generated_n,
            is_papercoder=True,
        )

    print(f"[DONE] round_root={round_root}")
    print(f"[DONE] baseline_artifacts={baseline_artifacts}")
    print(f"[DONE] baseline_repo={baseline_repo}")
    print(f"[DONE] feature_output_dir={feature_output_dir}")
    print(f"[DONE] output_repo_dir={output_repo_dir}")
    print(f"[DONE] baseline_interface_stub={stub_path}")


if __name__ == "__main__":
    setup_logging()
    main()
