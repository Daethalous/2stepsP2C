import argparse
import json
import os
from typing import Dict, List

from core.paper_router import build_stage_context


REQUIRED_SECTION_KEYS = ["id", "idx", "title", "level", "text", "labels", "confidence", "reason", "rescue"]


def _load_routing(output_dir: str) -> Dict:
    path = os.path.join(output_dir, "paper_section_routing.v2.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_sections(sections: List[Dict]) -> List[str]:
    errors = []
    for i, sec in enumerate(sections):
        missing = [k for k in REQUIRED_SECTION_KEYS if k not in sec]
        if missing:
            errors.append(f"section[{i}] missing keys: {missing}")
    return errors


def run_validate(output_dir: str, prompt_set: str) -> None:
    data = _load_routing(output_dir)
    sections = data.get("sections", [])
    errors = _validate_sections(sections)

    if errors:
        report = {"ok": False, "errors": errors}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    raw_chars = int(data.get("raw_text_chars", 0))
    stage_stats = {}
    for stage in ["planning", "analyzing", "coding"]:
        context, stats = build_stage_context(sections, prompt_set=prompt_set, stage=stage)
        stage_stats[stage] = {
            "context_chars": len(context),
            "selected_blocks": stats["selected_blocks"],
            "class_usage_chars": stats["class_usage_chars"],
            "reduction_vs_raw": round(1.0 - (len(context) / max(raw_chars, 1)), 4),
        }

    report = {
        "ok": True,
        "prompt_set": prompt_set,
        "routing_file": os.path.join(output_dir, "paper_section_routing.v2.json"),
        "raw_text_chars": raw_chars,
        "total_sections": len(sections),
        "stage_stats": stage_stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_set", type=str, default="baseline", choices=["baseline", "feature"])
    args = parser.parse_args()
    run_validate(args.output_dir, args.prompt_set)
