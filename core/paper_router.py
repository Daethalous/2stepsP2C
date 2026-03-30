import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.llm_engine import chat_completion_with_retry
from core.utils import parse_structured_json


Section = Dict[str, Any]
MainClassifierFn = Callable[[str, str], Optional[Dict[str, Any]]]
RescueClassifierFn = Callable[[str, str], Optional[Dict[str, Any]]]

TAG_BASELINE = "Baseline_Context"
TAG_FEATURE = "Feature_Context"
TAG_DISCARD = "Discard"
ALL_TAGS = [TAG_BASELINE, TAG_FEATURE, TAG_DISCARD]

STAGE_BUDGETS = {
    "planning": 24000,
    "analyzing": 22000,
    "coding": 22000,
}

STAGE_LABEL_PRIORITIES = {
    ("baseline", "planning"): [TAG_BASELINE, TAG_FEATURE],
    ("baseline", "analyzing"): [TAG_BASELINE, TAG_FEATURE],
    ("baseline", "coding"): [TAG_BASELINE, TAG_FEATURE],
    ("feature", "planning"): [TAG_FEATURE, TAG_BASELINE],
    ("feature", "analyzing"): [TAG_FEATURE, TAG_BASELINE],
    ("feature", "coding"): [TAG_FEATURE, TAG_BASELINE],
}

STAGE_LABEL_RATIOS = {
    ("baseline", "planning"): {TAG_BASELINE: 0.82, TAG_FEATURE: 0.18},
    ("baseline", "analyzing"): {TAG_BASELINE: 0.78, TAG_FEATURE: 0.22},
    ("baseline", "coding"): {TAG_BASELINE: 0.75, TAG_FEATURE: 0.25},
    ("feature", "planning"): {TAG_FEATURE: 0.78, TAG_BASELINE: 0.22},
    ("feature", "analyzing"): {TAG_FEATURE: 0.82, TAG_BASELINE: 0.18},
    ("feature", "coding"): {TAG_FEATURE: 0.82, TAG_BASELINE: 0.18},
}

BASELINE_HINTS = [
    "experiment", "evaluation", "dataset", "data preparation", "preprocess", "training setup",
    "validation", "test", "metric", "mse", "mae", "hyperparameter", "learning rate",
    "batch size", "epochs", "implementation details", "baseline",
]
FEATURE_HINTS = [
    "method", "methodology", "proposed model", "architecture", "algorithm", "objective",
    "loss", "proof", "theorem", "lemma", "optimization", "module", "innovation",
]
DISCARD_HINTS = [
    "related work", "conclusion", "future work", "acknowledgment", "acknowledgement",
    "reference", "bibliography", "limitations", "ethics statement",
]


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(title).lower())


def _slugify_title(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", _safe_text(title)).strip("_").lower()
    return slug or "untitled"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(text))


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:16]


def _dedup_lines(text: str) -> str:
    seen = set()
    out = []
    for line in _safe_text(text).splitlines():
        norm = re.sub(r"\s+", " ", line).strip()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(line.strip())
    return "\n".join(out)


def _extract_from_json_structured(paper_content: Any) -> List[Section]:
    if not isinstance(paper_content, (dict, list)):
        return []
    root = paper_content
    if isinstance(root, dict) and isinstance(root.get("pdf_parse"), dict):
        root = root["pdf_parse"]
    if not isinstance(root, dict):
        return []

    sections: List[Section] = []

    def add(title: str, text: str, level: int = 1) -> None:
        t = _dedup_lines(text)
        if not t:
            return
        sections.append({
            "title": _safe_text(title) or "Untitled",
            "level": int(level),
            "text": t,
        })

    abstract = root.get("abstract")
    if isinstance(abstract, list):
        parts = []
        for p in abstract:
            if isinstance(p, dict):
                parts.append(_safe_text(p.get("text", "")))
        add("Abstract", "\n".join([x for x in parts if x]), 1)
    elif isinstance(abstract, str):
        add("Abstract", abstract, 1)

    body = root.get("body_text")
    if isinstance(body, list):
        grouped: Dict[str, List[str]] = {}
        for para in body:
            if not isinstance(para, dict):
                continue
            sec_title = _safe_text(para.get("section")) or _safe_text(para.get("title")) or "Body"
            txt = _safe_text(para.get("text", ""))
            if txt:
                grouped.setdefault(sec_title, []).append(txt)
        for sec_title, chunks in grouped.items():
            add(sec_title, "\n".join(chunks), 1)
    return sections


def _extract_from_json_fallback(paper_content: Any) -> List[Section]:
    sections: List[Section] = []
    if not isinstance(paper_content, (dict, list)):
        return sections
    key_title_candidates = ["section", "title", "heading", "header", "name"]
    key_text_candidates = ["text", "content", "paragraph", "body"]

    def walk(node: Any, inherited_title: str = "") -> None:
        if isinstance(node, dict):
            title = inherited_title
            for k in key_title_candidates:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break
            found_text = ""
            for k in key_text_candidates:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    found_text = v.strip()
                    break
            if found_text:
                sections.append({"title": title or "Untitled", "level": 1, "text": _dedup_lines(found_text)})
            for v in node.values():
                walk(v, title)
        elif isinstance(node, list):
            for item in node:
                walk(item, inherited_title)

    walk(paper_content)
    return sections


def _extract_from_latex(latex_text: str) -> List[Section]:
    text = _safe_text(latex_text)
    if not text:
        return []
    pattern = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}")
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"title": "Full Text", "level": 1, "text": _dedup_lines(text)}]
    sections: List[Section] = []
    level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append({"title": "Preamble", "level": 1, "text": _dedup_lines(preamble)})
    for i, m in enumerate(matches):
        cmd = m.group(1)
        title = _safe_text(m.group(2)) or "Untitled"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _dedup_lines(text[start:end])
        if body:
            sections.append({"title": title, "level": level_map.get(cmd, 1), "text": body})
    return sections


def extract_sections(paper_content: Any, paper_format: str) -> List[Section]:
    raw_sections: List[Section]
    if paper_format == "LaTeX":
        raw_sections = _extract_from_latex(str(paper_content))
    else:
        raw_sections = _extract_from_json_structured(paper_content)
        if not raw_sections:
            raw_sections = _extract_from_json_fallback(paper_content)
        if not raw_sections:
            raw_sections = [{"title": "Full Text", "level": 1, "text": _safe_text(paper_content)}]

    merged_by_title: Dict[str, Dict[str, Any]] = {}
    for sec in raw_sections:
        title = _safe_text(sec.get("title")) or "Untitled"
        text = _dedup_lines(sec.get("text", ""))
        if not text:
            continue
        key = _slugify_title(title)
        if key not in merged_by_title:
            merged_by_title[key] = {"title": title, "level": int(sec.get("level", 1) or 1), "parts": []}
        merged_by_title[key]["parts"].append(text)

    out: List[Section] = []
    seen_hashes = set()
    for idx, (k, v) in enumerate(merged_by_title.items()):
        parts = []
        local_hashes = set()
        for p in v["parts"]:
            h = _text_hash(p)
            if h in local_hashes:
                continue
            local_hashes.add(h)
            parts.append(p)
        text = _dedup_lines("\n".join(parts))
        if not text:
            continue
        h = _text_hash(f"{k}:{text}")
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        out.append({
            "id": f"s{idx+1}_{k}",
            "idx": idx,
            "title": v["title"],
            "level": v["level"],
            "text": text,
            "text_hash": _text_hash(text),
        })
    return out


def _keyword_score(title: str, text: str, hints: List[str]) -> int:
    t = _normalize_title(title)
    b = _safe_text(text).lower()
    score = 0
    for kw in hints:
        if kw in t:
            score += 3
        if kw in b:
            score += 1
    return score


def _rule_main_classify(title: str, text: str) -> Dict[str, Any]:
    base_score = _keyword_score(title, text, BASELINE_HINTS)
    feat_score = _keyword_score(title, text, FEATURE_HINTS)
    discard_score = _keyword_score(title, text, DISCARD_HINTS)
    baseline_hit = base_score > 0
    feature_hit = feat_score > 0
    discard_candidate = discard_score > 0 and not baseline_hit and not feature_hit
    confidence = 0.5
    if baseline_hit or feature_hit:
        confidence = 0.65
    if discard_candidate:
        confidence = 0.55
    if not baseline_hit and not feature_hit and not discard_candidate:
        baseline_hit = True
        confidence = 0.4
    return {
        "baseline_hit": baseline_hit,
        "feature_hit": feature_hit,
        "discard_candidate": discard_candidate,
        "confidence": confidence,
        "evidence": [],
        "reason": f"rule:base={base_score},feat={feat_score},discard={discard_score}",
    }


def _build_labels(baseline_hit: bool, feature_hit: bool, discard_hit: bool) -> List[str]:
    if discard_hit and not baseline_hit and not feature_hit:
        return [TAG_DISCARD]
    labels = []
    if baseline_hit:
        labels.append(TAG_BASELINE)
    if feature_hit:
        labels.append(TAG_FEATURE)
    if not labels:
        labels = [TAG_BASELINE]
    return labels


def _llm_main_classify(client: Any, gpt_version: str, title: str, section_text: str) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You classify paper sections for code reproduction. "
                "Return ONLY [CONTENT]{json}[/CONTENT]. "
                "Schema: {"
                "\"baseline_hit\": bool, "
                "\"feature_hit\": bool, "
                "\"discard_candidate\": bool, "
                "\"confidence\": number, "
                "\"evidence\": [string], "
                "\"reason\": string"
                "}. "
                "Rules: allow baseline_hit and feature_hit to both be true. "
                "discard_candidate can be true ONLY if section is irrelevant to implementation. "
                "If uncertain, set discard_candidate=false."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Section title:\n{title}\n\n"
                f"Section content:\n{section_text[:7000]}"
            ),
        },
    ]
    try:
        completion = chat_completion_with_retry(client, gpt_version, messages)
        payload = parse_structured_json(completion.choices[0].message.content)
        if not isinstance(payload, dict):
            return None
        return {
            "baseline_hit": bool(payload.get("baseline_hit", False)),
            "feature_hit": bool(payload.get("feature_hit", False)),
            "discard_candidate": bool(payload.get("discard_candidate", False)),
            "confidence": float(payload.get("confidence", 0.7)),
            "evidence": payload.get("evidence", []) if isinstance(payload.get("evidence", []), list) else [],
            "reason": _safe_text(payload.get("reason", "llm_main")),
        }
    except Exception:
        return None


def _llm_rescue_discard(client: Any, gpt_version: str, title: str, section_text: str) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a recall guard for reproduction-critical paper routing. "
                "Input section is candidate Discard. "
                "Return ONLY [CONTENT]{json}[/CONTENT]. "
                "Schema: {"
                "\"should_rescue\": bool, "
                "\"rescued_to_baseline\": bool, "
                "\"rescued_to_feature\": bool, "
                "\"evidence\": [string], "
                "\"reason\": string"
                "}. "
                "If any possible implementation info exists, should_rescue must be true."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Section title:\n{title}\n\n"
                f"Section content:\n{section_text[:7000]}"
            ),
        },
    ]
    try:
        completion = chat_completion_with_retry(client, gpt_version, messages)
        payload = parse_structured_json(completion.choices[0].message.content)
        if not isinstance(payload, dict):
            return None
        return {
            "should_rescue": bool(payload.get("should_rescue", False)),
            "rescued_to_baseline": bool(payload.get("rescued_to_baseline", False)),
            "rescued_to_feature": bool(payload.get("rescued_to_feature", False)),
            "evidence": payload.get("evidence", []) if isinstance(payload.get("evidence", []), list) else [],
            "reason": _safe_text(payload.get("reason", "llm_rescue")),
        }
    except Exception:
        return None


def classify_sections_v2(
    sections: List[Section],
    client: Any = None,
    gpt_version: str = "gpt-5-mini",
    main_classifier: Optional[MainClassifierFn] = None,
    rescue_classifier: Optional[RescueClassifierFn] = None,
) -> List[Section]:
    out = []
    for sec in sections:
        title = sec.get("title", "")
        text = sec.get("text", "")
        pred = _rule_main_classify(title, text)

        if main_classifier:
            custom = main_classifier(title, text[:7000])
            if isinstance(custom, dict):
                pred.update({
                    "baseline_hit": bool(custom.get("baseline_hit", pred["baseline_hit"])),
                    "feature_hit": bool(custom.get("feature_hit", pred["feature_hit"])),
                    "discard_candidate": bool(custom.get("discard_candidate", pred["discard_candidate"])),
                    "confidence": float(custom.get("confidence", pred["confidence"])),
                    "evidence": custom.get("evidence", pred["evidence"]),
                    "reason": _safe_text(custom.get("reason", pred["reason"])),
                })
        elif client is not None:
            llm_pred = _llm_main_classify(client, gpt_version, title, text)
            if llm_pred:
                pred = llm_pred

        rescue_info = {
            "rescued": False,
            "rescued_to_baseline": False,
            "rescued_to_feature": False,
            "rescue_reason": "",
            "rescue_evidence": [],
        }
        if pred["discard_candidate"]:
            rescued = None
            if rescue_classifier:
                rescued = rescue_classifier(title, text[:7000])
            elif client is not None:
                rescued = _llm_rescue_discard(client, gpt_version, title, text)
            if isinstance(rescued, dict) and rescued.get("should_rescue", False):
                pred["discard_candidate"] = False
                pred["baseline_hit"] = bool(pred["baseline_hit"] or rescued.get("rescued_to_baseline", False))
                pred["feature_hit"] = bool(pred["feature_hit"] or rescued.get("rescued_to_feature", False))
                if not pred["baseline_hit"] and not pred["feature_hit"]:
                    pred["baseline_hit"] = True
                rescue_info = {
                    "rescued": True,
                    "rescued_to_baseline": bool(rescued.get("rescued_to_baseline", False)),
                    "rescued_to_feature": bool(rescued.get("rescued_to_feature", False)),
                    "rescue_reason": _safe_text(rescued.get("reason", "")),
                    "rescue_evidence": rescued.get("evidence", []) if isinstance(rescued.get("evidence", []), list) else [],
                }

        labels = _build_labels(
            baseline_hit=pred["baseline_hit"],
            feature_hit=pred["feature_hit"],
            discard_hit=pred["discard_candidate"],
        )
        enriched = dict(sec)
        enriched["labels"] = labels
        enriched["confidence"] = round(float(pred["confidence"]), 4)
        enriched["reason"] = pred["reason"]
        enriched["evidence"] = pred.get("evidence", []) if isinstance(pred.get("evidence", []), list) else []
        enriched["rescue"] = rescue_info
        out.append(enriched)
    return out


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated for token budget)..."


def _label_priority(prompt_set: str, stage: str) -> List[str]:
    return STAGE_LABEL_PRIORITIES.get((prompt_set.lower(), stage.lower()), [TAG_BASELINE, TAG_FEATURE])


def _label_ratio(prompt_set: str, stage: str) -> Dict[str, float]:
    return STAGE_LABEL_RATIOS.get((prompt_set.lower(), stage.lower()), {TAG_BASELINE: 0.5, TAG_FEATURE: 0.5})


def build_stage_context(
    classified_sections: List[Section],
    prompt_set: str,
    stage: str,
    max_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    prompt_set = (prompt_set or "baseline").lower()
    stage = stage.lower()
    max_chars = max_chars or STAGE_BUDGETS.get(stage, 22000)
    priorities = _label_priority(prompt_set, stage)
    ratios = _label_ratio(prompt_set, stage)
    caps = {k: int(max_chars * ratios.get(k, 0)) for k in [TAG_BASELINE, TAG_FEATURE]}
    usage = {TAG_BASELINE: 0, TAG_FEATURE: 0}

    selectable = []
    for sec in classified_sections:
        labels = sec.get("labels", [])
        if TAG_DISCARD in labels:
            continue
        if TAG_BASELINE not in labels and TAG_FEATURE not in labels:
            continue
        rank = 99
        for i, tag in enumerate(priorities):
            if tag in labels:
                rank = min(rank, i)
        selectable.append((rank, sec))
    selectable.sort(key=lambda x: (x[0], -float(x[1].get("confidence", 0.0)), len(x[1].get("text", ""))))

    blocks = []
    total_chars = 0
    selected_ids = []
    for _, sec in selectable:
        labels = [t for t in sec.get("labels", []) if t in (TAG_BASELINE, TAG_FEATURE)]
        if not labels:
            continue
        remaining = max_chars - total_chars
        if remaining <= 200:
            break
        label_allowance = remaining
        for tag in labels:
            tag_remaining = max(0, caps.get(tag, 0) - usage.get(tag, 0))
            if tag_remaining > 0:
                label_allowance = min(label_allowance, tag_remaining)
        if label_allowance <= 200:
            continue
        body = _clip_text(sec.get("text", ""), label_allowance)
        block = f"## [{','.join(labels)}] {sec.get('title','Untitled')}\n{body}\n"
        if total_chars + len(block) > max_chars:
            continue
        blocks.append(block)
        total_chars += len(block)
        selected_ids.append(sec.get("id", ""))
        for tag in labels:
            usage[tag] += len(body)

    if not blocks:
        fallback_text = _clip_text(
            "\n\n".join([s.get("text", "") for s in classified_sections[:2]]),
            max(200, max_chars - 180),
        )
        blocks = [f"## [{TAG_BASELINE}] Fallback\n{fallback_text}\n"]

    header = (
        f"[PaperRoutingV2] prompt_set={prompt_set} stage={stage}\n"
        f"[PaperRoutingV2] labels={','.join(priorities)} budget={max_chars}\n"
    )
    context = header + "\n".join(blocks)

    label_counts = {
        TAG_BASELINE: sum(1 for s in classified_sections if TAG_BASELINE in s.get("labels", [])),
        TAG_FEATURE: sum(1 for s in classified_sections if TAG_FEATURE in s.get("labels", [])),
        TAG_DISCARD: sum(1 for s in classified_sections if TAG_DISCARD in s.get("labels", [])),
    }
    rescued = sum(1 for s in classified_sections if bool(s.get("rescue", {}).get("rescued", False)))
    stats = {
        "prompt_set": prompt_set,
        "stage": stage,
        "labels": priorities,
        "max_chars": max_chars,
        "selected_blocks": len(blocks),
        "selected_ids": selected_ids,
        "context_chars": len(context),
        "class_usage_chars": usage,
        "label_counts": label_counts,
        "rescued_count": rescued,
    }
    return context, stats


def _routing_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "paper_section_routing.v2.json")


def _legacy_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "paper_section_routing.json")


def _serialize_routing(sections: List[Section], paper_format: str) -> Dict[str, Any]:
    return {
        "version": 2,
        "paper_format": paper_format,
        "total_sections": len(sections),
        "raw_text_chars": sum(len(s.get("text", "")) for s in sections),
        "sections": sections,
    }


def _load_cached_routing(output_dir: str) -> Optional[Dict[str, Any]]:
    v2_path = _routing_cache_path(output_dir)
    if os.path.exists(v2_path):
        try:
            with open(v2_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("sections"), list):
                return data
        except Exception:
            pass
    # 兼容旧缓存：不复用旧标签体系，显式返回None触发重算
    if os.path.exists(_legacy_cache_path(output_dir)):
        return None
    return None


def _save_cached_routing(output_dir: str, data: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(_routing_cache_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def prepare_paper_context_for_stage(
    paper_content: Any,
    paper_format: str,
    prompt_set: str,
    stage: str,
    output_dir: str,
    max_chars: Optional[int] = None,
    client: Any = None,
    gpt_version: str = "gpt-5-mini",
    main_classifier: Optional[MainClassifierFn] = None,
    rescue_classifier: Optional[RescueClassifierFn] = None,
    force_rebuild: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if client is None and main_classifier is None:
        raise ValueError(
            "Paper routing requires LLM classification. "
            "Provide OPENAI_API_KEY (client) or pass custom classifiers."
        )
    cached = None if force_rebuild else _load_cached_routing(output_dir)
    if cached:
        sections = cached["sections"]
    else:
        sections = extract_sections(paper_content, paper_format)
        sections = classify_sections_v2(
            sections=sections,
            client=client,
            gpt_version=gpt_version,
            main_classifier=main_classifier,
            rescue_classifier=rescue_classifier,
        )
        cached = _serialize_routing(sections, paper_format)
        _save_cached_routing(output_dir, cached)

    context, stats = build_stage_context(
        classified_sections=sections,
        prompt_set=(prompt_set or "baseline"),
        stage=stage,
        max_chars=max_chars,
    )
    stats["total_sections"] = len(sections)
    stats["routing_cache"] = _routing_cache_path(output_dir)
    stats["raw_text_chars"] = cached.get("raw_text_chars", 0)
    return context, stats


def export_agent_contexts(
    paper_content: Any,
    paper_format: str,
    output_dir: str,
    stage: str = "planning",
    baseline_max_chars: Optional[int] = None,
    feature_max_chars: Optional[int] = None,
    client: Any = None,
    gpt_version: str = "gpt-5-mini",
    main_classifier: Optional[MainClassifierFn] = None,
    rescue_classifier: Optional[RescueClassifierFn] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    baseline_context, baseline_stats = prepare_paper_context_for_stage(
        paper_content=paper_content,
        paper_format=paper_format,
        prompt_set="baseline",
        stage=stage,
        output_dir=output_dir,
        max_chars=baseline_max_chars,
        client=client,
        gpt_version=gpt_version,
        main_classifier=main_classifier,
        rescue_classifier=rescue_classifier,
        force_rebuild=force_rebuild,
    )
    feature_context, feature_stats = prepare_paper_context_for_stage(
        paper_content=paper_content,
        paper_format=paper_format,
        prompt_set="feature",
        stage=stage,
        output_dir=output_dir,
        max_chars=feature_max_chars,
        client=client,
        gpt_version=gpt_version,
        main_classifier=main_classifier,
        rescue_classifier=rescue_classifier,
        force_rebuild=False,
    )

    baseline_path = os.path.join(output_dir, "paper_context_baseline.txt")
    feature_path = os.path.join(output_dir, "paper_context_feature.txt")
    manifest_path = os.path.join(output_dir, "paper_contexts_manifest.json")

    os.makedirs(output_dir, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        f.write(baseline_context)
    with open(feature_path, "w", encoding="utf-8") as f:
        f.write(feature_context)

    manifest = {
        "version": 1,
        "stage": stage,
        "routing_cache": _routing_cache_path(output_dir),
        "paper_format": paper_format,
        "contexts": {
            "baseline": {
                "path": baseline_path,
                "chars": len(baseline_context),
                "stats": baseline_stats,
            },
            "feature": {
                "path": feature_path,
                "chars": len(feature_context),
                "stats": feature_stats,
            },
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "baseline_context": baseline_context,
        "feature_context": feature_context,
        "baseline_stats": baseline_stats,
        "feature_stats": feature_stats,
        "baseline_path": baseline_path,
        "feature_path": feature_path,
        "manifest_path": manifest_path,
    }
