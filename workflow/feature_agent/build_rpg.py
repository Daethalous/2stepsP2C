import argparse
import json
import os
from typing import Any, Dict, List, Optional

from core.data_loader import load_pipeline_context, sanitize_todo_file_name
from core.logger import get_logger
from workflow.baseline_agent.build_rpg import PipelineRPG

logger = get_logger(__name__)

FEATURE_RPG_FILENAME = "feature_rpg_graph.json"


def _normalize_rel_path(path_text: Any) -> str:
    if path_text is None:
        return ""
    return sanitize_todo_file_name(str(path_text)) or str(path_text).strip().replace("\\", "/")


def _derive_baseline_output_dir(baseline_interface_stub_path: Optional[str]) -> str:
    if not baseline_interface_stub_path:
        return ""
    candidate = os.path.abspath(baseline_interface_stub_path)
    if os.path.isdir(candidate):
        return candidate
    return os.path.dirname(candidate)


def _load_baseline_rpg(baseline_output_dir: str) -> Optional[PipelineRPG]:
    if not baseline_output_dir:
        return None
    rpg_path = os.path.join(baseline_output_dir, "rpg_graph.json")
    if not os.path.exists(rpg_path):
        return None
    try:
        return PipelineRPG.load(rpg_path)
    except Exception as exc:
        logger.warning(f"[Feature RPG] Failed to load baseline RPG from {rpg_path}: {exc}")
        return None


def _dedupe_paths(values: List[Any], node_set: set[str], skip: Optional[set[str]] = None) -> List[str]:
    result: List[str] = []
    skip = skip or set()
    for value in values:
        path = _normalize_rel_path(value)
        if not path or path in skip or path not in node_set or path in result:
            continue
        result.append(path)
    return result


def _merge_feature_edges(
    rpg: PipelineRPG,
    target_file: str,
    node_set: set[str],
    closure: Dict[str, Any],
    context_bundle: Dict[str, Any],
    baseline_rpg: Optional[PipelineRPG],
) -> Dict[str, List[str]]:
    upstream = _dedupe_paths(context_bundle.get("upstream_caller_files", []), node_set, {target_file})
    downstream = _dedupe_paths(context_bundle.get("downstream_callee_files", []), node_set, {target_file})
    shared = _dedupe_paths(context_bundle.get("shared_interface_files", []), node_set, {target_file})
    config_registry = _dedupe_paths(context_bundle.get("config_and_registry_files", []), node_set, {target_file})
    optional_related = _dedupe_paths(context_bundle.get("optional_related_files", []), node_set, {target_file})
    required_context = _dedupe_paths(closure.get("required_context_files", []), node_set, {target_file})
    synchronized_edits = _dedupe_paths(
        closure.get("synchronized_edits", []) or context_bundle.get("synchronized_edit_targets", []),
        node_set,
        {target_file},
    )

    baseline_deps: List[str] = []
    if baseline_rpg and target_file in baseline_rpg.nodes:
        baseline_deps = _dedupe_paths(baseline_rpg.get_dependencies(target_file), node_set, {target_file})

    reverse_dependencies = set(upstream) | set(synchronized_edits)
    dependency_candidates = []
    for group in (downstream, shared, config_registry, baseline_deps, required_context):
        for dep in group:
            if dep not in reverse_dependencies and dep not in dependency_candidates:
                dependency_candidates.append(dep)

    for dep in dependency_candidates:
        rpg.add_dependency(target_file, dep)

    for caller in upstream + synchronized_edits:
        rpg.add_dependency(caller, target_file)

    return {
        "required_context_files": required_context,
        "upstream_callers": upstream,
        "downstream_callees": downstream,
        "shared_interface_files": shared,
        "config_and_registry_files": config_registry,
        "optional_related_files": optional_related,
        "synchronized_edits": synchronized_edits,
        "baseline_dependencies": baseline_deps,
    }


def build_feature_rpg(output_dir: str, baseline_interface_stub_path: Optional[str] = None) -> PipelineRPG:
    ctx = load_pipeline_context(output_dir)
    todo_file_lst = [_normalize_rel_path(x) for x in ctx.todo_file_lst if _normalize_rel_path(x)]
    node_set = set(todo_file_lst)
    baseline_output_dir = _derive_baseline_output_dir(baseline_interface_stub_path)
    baseline_rpg = _load_baseline_rpg(baseline_output_dir)

    rpg = PipelineRPG(repo_name="feature_incremental_rpg")
    file_metadata: Dict[str, Dict[str, Any]] = {}

    for todo_file_name in todo_file_lst:
        closure = ctx.modification_closure_by_file.get(todo_file_name, {}) or {}
        role_tags = list(closure.get("focus_role_tags", []))
        rpg.add_file(
            todo_file_name,
            description=str(ctx.logic_analysis_dict.get(todo_file_name, "")),
            features=role_tags,
        )

    for todo_file_name in todo_file_lst:
        closure = ctx.modification_closure_by_file.get(todo_file_name, {}) or {}
        context_bundle = ctx.context_bundle_by_file.get(todo_file_name, {}) or {}
        edge_info = _merge_feature_edges(
            rpg=rpg,
            target_file=todo_file_name,
            node_set=node_set,
            closure=closure,
            context_bundle=context_bundle,
            baseline_rpg=baseline_rpg,
        )
        file_metadata[todo_file_name] = {
            "entrypoint_chain": list(context_bundle.get("entrypoint_chain", []) or closure.get("entrypoints", [])),
            "synchronized_edits": edge_info["synchronized_edits"],
            "target_symbols": list(context_bundle.get("target_symbols", []) or closure.get("target_symbols", [])),
            "focus_role_tags": list(closure.get("focus_role_tags", [])),
            "required_context_files": edge_info["required_context_files"],
            "upstream_callers": edge_info["upstream_callers"],
            "downstream_callees": edge_info["downstream_callees"],
            "shared_interface_files": edge_info["shared_interface_files"],
            "config_and_registry_files": edge_info["config_and_registry_files"],
            "optional_related_files": edge_info["optional_related_files"],
            "baseline_dependencies": edge_info["baseline_dependencies"],
        }

    graph_path = os.path.join(output_dir, FEATURE_RPG_FILENAME)
    payload = rpg.to_dict()
    payload["file_metadata"] = file_metadata
    payload["source"] = "feature_rpg_v1"
    payload["baseline_output_dir"] = baseline_output_dir
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[Feature RPG] Saved {len(rpg.nodes)} nodes and {len(rpg.edges)} edges to {graph_path}"
    )
    return rpg


def run_build_feature_rpg(output_dir: str, baseline_interface_stub_path: Optional[str] = None) -> None:
    build_feature_rpg(
        output_dir=output_dir,
        baseline_interface_stub_path=baseline_interface_stub_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature RPG graph from feature planning artifacts.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_interface_stub", type=str, default="")
    args = parser.parse_args()
    run_build_feature_rpg(
        output_dir=args.output_dir,
        baseline_interface_stub_path=args.baseline_interface_stub or None,
    )
