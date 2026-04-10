import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.data_loader import sanitize_todo_file_name
from workflow.baseline_agent.build_rpg import PipelineRPG
from workflow.baseline_agent.rpg_adapter import (
    get_coding_context as _get_baseline_coding_context,
    get_stub_context as _get_baseline_stub_context,
    load_stubs_dict as _load_baseline_stubs_dict,
)

FEATURE_RPG_FILENAME = "feature_rpg_graph.json"


def make_safe_artifact_stem(path_text: str) -> str:
    normalized = sanitize_todo_file_name(path_text) or str(path_text or "").strip()
    normalized = normalized.replace("\\", "/")
    readable = normalized.replace("/", "__")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("._")
    readable = re.sub(r"_+", "_", readable)
    if not readable:
        readable = "artifact"
    import hashlib
    suffix = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def derive_baseline_output_dir(baseline_interface_stub_path: Optional[str]) -> str:
    if not baseline_interface_stub_path:
        return ""
    candidate = os.path.abspath(baseline_interface_stub_path)
    if os.path.isdir(candidate):
        return candidate
    return os.path.dirname(candidate)


def load_or_build_feature_rpg_bundle(
    output_dir: str,
    baseline_interface_stub_path: Optional[str] = None,
) -> Tuple[PipelineRPG, Dict[str, Dict[str, Any]]]:
    graph_path = os.path.join(output_dir, FEATURE_RPG_FILENAME)
    if not os.path.exists(graph_path):
        from workflow.feature_agent.build_rpg import build_feature_rpg

        build_feature_rpg(
            output_dir=output_dir,
            baseline_interface_stub_path=baseline_interface_stub_path,
        )
    with open(graph_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return PipelineRPG.from_dict(payload), payload.get("file_metadata", {})


def _compute_depths(rpg: PipelineRPG) -> Dict[str, int]:
    memo: Dict[str, int] = {}

    def _depth(node: str, stack: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in stack:
            return 0
        stack.add(node)
        deps = rpg.get_dependencies(node)
        if not deps:
            depth = 0
        else:
            depth = 1 + max(_depth(dep, stack) for dep in deps)
        stack.remove(node)
        memo[node] = depth
        return depth

    for path in rpg.nodes:
        _depth(path, set())
    return memo


def _role_priority(path_name: str, file_meta: Dict[str, Any]) -> int:
    role_tags = list(file_meta.get("focus_role_tags", []))
    if any(tag in role_tags for tag in ("config",)):
        return 0
    if file_meta.get("target_symbols"):
        return 1
    if any(tag in role_tags for tag in ("registry", "factory", "entrypoint", "main", "train", "eval")):
        return 3
    if path_name.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh")):
        return 4
    return 2


def get_feature_file_order(
    rpg: PipelineRPG,
    original_order: List[str],
    file_metadata: Dict[str, Dict[str, Any]],
) -> List[str]:
    normalized_original = [sanitize_todo_file_name(x) or x for x in original_order]
    original_index = {path: idx for idx, path in enumerate(normalized_original)}
    topo = rpg.topological_sort()
    topo_index = {path: idx for idx, path in enumerate(topo)}
    depths = _compute_depths(rpg)
    ordered = sorted(
        [path for path in topo if path in original_index],
        key=lambda path: (
            depths.get(path, 0),
            _role_priority(path, file_metadata.get(path, {})),
            topo_index.get(path, 10**6),
            original_index.get(path, 10**6),
            path,
        ),
    )
    seen = set(ordered)
    for path in normalized_original:
        if path not in seen:
            ordered.append(path)
    return ordered


def _load_analysis_text(base_dir: str, todo_file_name: str) -> str:
    if not base_dir:
        return ""
    candidate = sanitize_todo_file_name(todo_file_name) or todo_file_name
    safe_name = make_safe_artifact_stem(candidate)
    hashed_path = os.path.join(base_dir, "analyzing_artifacts", f"{safe_name}_simple_analysis.txt")
    legacy_path = os.path.join(base_dir, "analyzing_artifacts", f"{candidate.replace('/', '_')}_simple_analysis.txt")
    for path in (hashed_path, legacy_path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def load_baseline_analysis_dict(
    baseline_interface_stub_path: Optional[str],
    todo_file_lst: Optional[List[str]] = None,
) -> Dict[str, str]:
    baseline_output_dir = derive_baseline_output_dir(baseline_interface_stub_path)
    if not baseline_output_dir or not todo_file_lst:
        return {}
    result: Dict[str, str] = {}
    for todo_file_name in todo_file_lst:
        text = _load_analysis_text(baseline_output_dir, todo_file_name)
        if text:
            result[todo_file_name] = text
    return result


def get_feature_analysis_context(
    rpg: PipelineRPG,
    target_file: str,
    completed_analysis_dict: Dict[str, str],
    baseline_analysis_dict: Dict[str, str],
    max_chars: int = 4000,
) -> str:
    deps = rpg.get_dependencies(target_file)
    if not deps:
        return ""
    parts = [f"## Upstream Dependencies for {target_file}\n"]
    total = len(parts[0])
    for dep in deps:
        analysis = completed_analysis_dict.get(dep) or baseline_analysis_dict.get(dep, "")
        if not analysis:
            continue
        truncated = analysis[:800]
        if len(analysis) > 800:
            truncated += "...(truncated)"
        chunk = f"\n### {dep}\n{truncated}\n"
        if total + len(chunk) > max_chars:
            parts.append("\n...(remaining upstream analyses truncated)...\n")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts) if len(parts) > 1 else ""


def load_verified_baseline_stubs(baseline_interface_stub_path: Optional[str]) -> Dict[str, str]:
    baseline_output_dir = derive_baseline_output_dir(baseline_interface_stub_path)
    if not baseline_output_dir:
        return {}
    return _load_baseline_stubs_dict(baseline_output_dir)


def get_verified_stub_context(
    stubs_dict: Dict[str, str],
    target_file: str,
    rpg: PipelineRPG,
) -> Tuple[str, str]:
    return _get_baseline_stub_context(stubs_dict, target_file, rpg)


def build_feature_coding_context(
    rpg: PipelineRPG,
    target_file: str,
    done_file_dict: Dict[str, str],
    max_total_chars: int = 14000,
) -> str:
    return _get_baseline_coding_context(rpg, target_file, done_file_dict, max_total_chars)
