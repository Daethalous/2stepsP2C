import ast
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional


REPO_INDEX_FILENAMES = {
    "repo_manifest": "repo_manifest.json",
    "symbol_index": "symbol_index.json",
    "call_graph": "call_graph.json",
    "entrypoint_index": "entrypoint_index.json",
    "modification_closure": "modification_closure.json",
    "context_bundle": "context_bundle.json",
}

SUPPORTED_REPO_EXTENSIONS = (
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
    ".sh",
)

CONFIG_EXTENSIONS = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")
PROMPT_RELEVANT_EXTENSIONS = (".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh", ".txt", ".md")
SKIP_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".cursor",
    "outputs",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}
MAX_INDEX_FILE_BYTES = 512 * 1024


def _normalize_rel_path(path_text: str) -> str:
    return str(path_text or "").replace("\\", "/").strip().lstrip("./")


def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _safe_json_dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _iter_repo_files(repo_dir: str) -> List[str]:
    rel_paths: List[str] = []
    if not repo_dir or not os.path.isdir(repo_dir):
        return rel_paths
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRECTORIES]
        for file_name in sorted(files):
            if file_name.startswith("."):
                continue
            rel_path = _normalize_rel_path(os.path.relpath(os.path.join(root, file_name), repo_dir))
            if os.path.splitext(rel_path)[1].lower() not in SUPPORTED_REPO_EXTENSIONS:
                continue
            abs_path = os.path.join(root, file_name)
            try:
                if os.path.getsize(abs_path) > MAX_INDEX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel_paths.append(rel_path)
    return sorted(rel_paths)


def _build_file_role_tags(rel_path: str, content: str) -> List[str]:
    lower_path = rel_path.lower()
    lower_text = content.lower()
    tags = set()
    if rel_path.endswith(CONFIG_EXTENSIONS):
        tags.add("config")
    if rel_path.endswith(".md"):
        tags.add("notes")
    if rel_path.endswith(".txt"):
        tags.add("text")
    if rel_path.endswith(".sh"):
        tags.add("script")
    if re.search(r"(^|/)(test|tests|testing)/", lower_path) or lower_path.startswith("test"):
        tags.add("test")
    if lower_path.endswith("main.py") or lower_path.endswith("__main__.py"):
        tags.add("entrypoint")
        tags.add("main")
    if "__name__ == \"__main__\"" in content or "__name__ == '__main__'" in content:
        tags.add("entrypoint")
    for token, tag in (
        ("argparse", "cli"),
        ("click.", "cli"),
        ("def main(", "main"),
        ("train", "train"),
        ("trainer", "train"),
        ("eval", "eval"),
        ("evaluator", "eval"),
        ("infer", "inference"),
        ("predict", "inference"),
        ("registry", "registry"),
        ("register(", "registry"),
        ("factory", "factory"),
        ("config", "config"),
    ):
        if token in lower_path or token in lower_text:
            tags.add(tag)
    return sorted(tags)


def _signature_from_source(code_text: str, node: ast.AST, fallback_name: str) -> str:
    try:
        source = ast.get_source_segment(code_text, node) or ""
    except Exception:
        source = ""
    if source:
        first_line = source.splitlines()[0].strip()
        return first_line[:240]
    return fallback_name


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _dotted_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


def _module_to_candidates(module_name: str) -> List[str]:
    module_path = module_name.replace(".", "/")
    return [f"{module_path}.py", f"{module_path}/__init__.py"]


def _resolve_import_target(current_file: str, module_name: str, repo_files: set, level: int = 0) -> Optional[str]:
    current_parts = current_file.split("/")[:-1]
    if level > 0:
        anchor = current_parts[: max(len(current_parts) - level + 1, 0)]
        if module_name:
            anchor = anchor + module_name.split(".")
        candidates = ["/".join(anchor) + ".py", "/".join(anchor) + "/__init__.py"]
    else:
        candidates = _module_to_candidates(module_name) if module_name else []
    for candidate in candidates:
        candidate = candidate.strip("/")
        if candidate in repo_files:
            return candidate
    return None


def _analyze_python_file(rel_path: str, code_text: str, repo_files: set) -> dict:
    result = {
        "symbols": [],
        "imports": [],
        "call_names": [],
        "main_blocks": [],
        "parse_error": "",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code_text)
    except SyntaxError as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            result["symbols"].append(
                {
                    "name": node.name,
                    "kind": "function",
                    "lineno": getattr(node, "lineno", 0),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                    "signature": _signature_from_source(code_text, node, f"def {node.name}(...)"),
                }
            )
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            result["symbols"].append(
                {
                    "name": node.name,
                    "kind": "async_function",
                    "lineno": getattr(node, "lineno", 0),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                    "signature": _signature_from_source(code_text, node, f"async def {node.name}(...)"),
                }
            )
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            result["symbols"].append(
                {
                    "name": node.name,
                    "kind": "class",
                    "lineno": getattr(node, "lineno", 0),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                    "signature": _signature_from_source(code_text, node, f"class {node.name}"),
                }
            )
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                module_name = alias.name
                result["imports"].append(
                    {
                        "module": module_name,
                        "imported_symbol": "",
                        "alias": alias.asname or "",
                        "lineno": getattr(node, "lineno", 0),
                        "resolved_file": _resolve_import_target(rel_path, module_name, repo_files),
                    }
                )
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module_name = node.module or ""
            resolved = _resolve_import_target(rel_path, module_name, repo_files, level=node.level)
            for alias in node.names:
                result["imports"].append(
                    {
                        "module": module_name,
                        "imported_symbol": alias.name,
                        "alias": alias.asname or "",
                        "lineno": getattr(node, "lineno", 0),
                        "level": node.level,
                        "resolved_file": resolved,
                    }
                )
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            call_name = _dotted_name(node.func)
            if call_name:
                result["call_names"].append(call_name)
            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            test_text = ""
            try:
                test_text = ast.get_source_segment(code_text, node.test) or ""
            except Exception:
                test_text = ""
            if "__name__" in test_text and "__main__" in test_text:
                result["main_blocks"].append(
                    {
                        "lineno": getattr(node, "lineno", 0),
                        "test": test_text.strip(),
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    result["call_names"] = sorted(set(result["call_names"]))
    return result


def build_repo_index(repo_dir: str) -> Dict[str, Any]:
    repo_files = _iter_repo_files(repo_dir)
    repo_set = set(repo_files)
    manifest_files: Dict[str, Any] = {}
    symbol_files: Dict[str, Any] = {}
    symbol_to_files: Dict[str, List[str]] = defaultdict(list)
    edges: List[Dict[str, str]] = []

    for rel_path in repo_files:
        abs_path = os.path.join(repo_dir, rel_path)
        content = _safe_read_text(abs_path)
        ext = os.path.splitext(rel_path)[1].lower()
        role_tags = _build_file_role_tags(rel_path, content)
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
        manifest_files[rel_path] = {
            "path": rel_path,
            "ext": ext,
            "size": len(content),
            "sha1": digest,
            "role_tags": role_tags,
            "is_entry_candidate": "entrypoint" in role_tags or "main" in role_tags or "cli" in role_tags,
            "is_config": "config" in role_tags,
            "is_test": "test" in role_tags,
        }
        if ext == ".py":
            analysis = _analyze_python_file(rel_path, content, repo_set)
            symbol_files[rel_path] = analysis
            for item in analysis["symbols"]:
                symbol_to_files[item["name"]].append(rel_path)
            for imp in analysis["imports"]:
                if imp.get("resolved_file"):
                    edges.append(
                        {
                            "src": rel_path,
                            "dst": imp["resolved_file"],
                            "reason": "import",
                            "symbol": imp.get("imported_symbol", ""),
                        }
                    )

    for rel_path, analysis in symbol_files.items():
        for call_name in analysis.get("call_names", []):
            leaf = call_name.split(".")[-1]
            for candidate in symbol_to_files.get(leaf, []):
                if candidate == rel_path:
                    continue
                edges.append(
                    {
                        "src": rel_path,
                        "dst": candidate,
                        "reason": "symbol_call",
                        "symbol": leaf,
                    }
                )

    deduped_edges: List[Dict[str, str]] = []
    seen_edges = set()
    for edge in edges:
        key = (edge["src"], edge["dst"], edge["reason"], edge.get("symbol", ""))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_edges.append(edge)

    file_neighbors: Dict[str, Dict[str, List[str]]] = {}
    reverse_neighbors: Dict[str, List[str]] = defaultdict(list)
    for rel_path in symbol_files:
        file_neighbors[rel_path] = {"imports": [], "calls": []}
    for edge in deduped_edges:
        file_neighbors.setdefault(edge["src"], {"imports": [], "calls": []})
        bucket = "imports" if edge["reason"] == "import" else "calls"
        if edge["dst"] not in file_neighbors[edge["src"]][bucket]:
            file_neighbors[edge["src"]][bucket].append(edge["dst"])
        if edge["src"] not in reverse_neighbors[edge["dst"]]:
            reverse_neighbors[edge["dst"]].append(edge["src"])

    candidates = []
    for rel_path, meta in manifest_files.items():
        if rel_path.endswith(".py") and meta["is_entry_candidate"]:
            score = 0
            reasons = list(meta["role_tags"])
            if rel_path.endswith(("main.py", "__main__.py")):
                score += 5
            if "entrypoint" in meta["role_tags"]:
                score += 4
            if "cli" in meta["role_tags"]:
                score += 2
            if "train" in meta["role_tags"] or "eval" in meta["role_tags"]:
                score += 1
            if symbol_files.get(rel_path, {}).get("main_blocks"):
                score += 4
            candidates.append({"path": rel_path, "score": score, "reasons": sorted(set(reasons))})

    candidates.sort(key=lambda x: (-x["score"], x["path"]))
    primary_files = [item["path"] for item in candidates[:8]]
    execution_chains = []
    for root in primary_files[:4]:
        visited = set()
        queue = deque([(root, [root])])
        while queue:
            current, chain = queue.popleft()
            if current in visited or len(chain) > 6:
                continue
            visited.add(current)
            execution_chains.append(chain)
            for nxt in file_neighbors.get(current, {}).get("imports", [])[:12]:
                if nxt not in chain:
                    queue.append((nxt, chain + [nxt]))

    repo_index = {
        "repo_manifest": {
            "repo_dir": repo_dir,
            "files": manifest_files,
        },
        "symbol_index": {
            "files": symbol_files,
            "symbol_to_files": {key: sorted(set(value)) for key, value in symbol_to_files.items()},
        },
        "call_graph": {
            "edges": deduped_edges,
            "file_neighbors": file_neighbors,
            "reverse_neighbors": dict(reverse_neighbors),
        },
        "entrypoint_index": {
            "candidates": candidates,
            "primary_files": primary_files,
            "execution_chains": execution_chains,
        },
    }
    return repo_index


def save_repo_index(output_dir: str, repo_index: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, file_name in REPO_INDEX_FILENAMES.items():
        if key in repo_index and isinstance(repo_index[key], dict):
            _safe_json_dump(os.path.join(output_dir, file_name), repo_index[key])


def load_repo_index(output_dir: str) -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    for key, file_name in REPO_INDEX_FILENAMES.items():
        path = os.path.join(output_dir, file_name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded[key] = json.load(f)
        except Exception:
            loaded[key] = {}
    return loaded


def summarize_repo_index(repo_index: Dict[str, Any], max_files: int = 40, max_chars: int = 16000) -> str:
    manifest = repo_index.get("repo_manifest", {}).get("files", {})
    symbol_files = repo_index.get("symbol_index", {}).get("files", {})
    entrypoints = repo_index.get("entrypoint_index", {}).get("candidates", [])
    lines = ["Repository map:"]
    if entrypoints:
        lines.append("Primary entrypoint candidates:")
        for item in entrypoints[:6]:
            reasons = ", ".join(item.get("reasons", [])[:6])
            lines.append(f"- {item['path']} (score={item.get('score', 0)}; {reasons})")
    for rel_path in sorted(manifest)[:max_files]:
        meta = manifest[rel_path]
        role_tags = ",".join(meta.get("role_tags", [])[:6]) or "plain"
        symbols = symbol_files.get(rel_path, {}).get("symbols", [])
        symbol_names = ", ".join(item["name"] for item in symbols[:6]) or "(no symbols)"
        lines.append(f"- {rel_path} [{role_tags}] symbols: {symbol_names}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated for token budget)..."
    return text or "(none)"


def summarize_entrypoint_index(repo_index: Dict[str, Any], max_chars: int = 4000) -> str:
    entry_index = repo_index.get("entrypoint_index", {})
    candidates = entry_index.get("candidates", [])
    chains = entry_index.get("execution_chains", [])
    lines = ["Entrypoint summary:"]
    for item in candidates[:8]:
        reasons = ", ".join(item.get("reasons", [])[:6])
        lines.append(f"- {item['path']} -> {reasons}")
    if chains:
        lines.append("Execution chains:")
        seen = set()
        for chain in chains[:12]:
            chain_text = " -> ".join(chain)
            if chain_text in seen:
                continue
            seen.add(chain_text)
            lines.append(f"- {chain_text}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated for token budget)..."
    return text or "(none)"


def _extract_planned_symbols(raw_value: Any) -> List[str]:
    symbols: List[str] = []
    if isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    symbols.append(text)
    elif isinstance(raw_value, str):
        for chunk in re.split(r"[,;|]", raw_value):
            text = chunk.strip()
            if text:
                symbols.append(text)
    return symbols


def _extract_path_token(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    text = text.split("::", 1)[0].strip()
    match = re.search(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9._-]+", text)
    return _normalize_rel_path(match.group(0)) if match else ""


def _reverse_reachable(reverse_neighbors: Dict[str, List[str]], start: str, max_depth: int = 3) -> List[str]:
    visited = set()
    queue = deque([(start, 0)])
    ordered: List[str] = []
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for parent in reverse_neighbors.get(current, []):
            if parent in visited:
                continue
            visited.add(parent)
            ordered.append(parent)
            queue.append((parent, depth + 1))
    return ordered


def _forward_reachable(file_neighbors: Dict[str, Dict[str, List[str]]], start: str, max_depth: int = 3) -> List[str]:
    visited = set()
    queue = deque([(start, 0)])
    ordered: List[str] = []
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        neighbors = file_neighbors.get(current, {})
        for nxt in neighbors.get("imports", []) + neighbors.get("calls", []):
            if nxt in visited:
                continue
            visited.add(nxt)
            ordered.append(nxt)
            queue.append((nxt, depth + 1))
    return ordered


def _find_entrypoints_for_file(repo_index: Dict[str, Any], target_file: str) -> List[str]:
    target_file = _normalize_rel_path(target_file)
    entry_index = repo_index.get("entrypoint_index", {})
    primary_files = entry_index.get("primary_files", [])
    reverse_neighbors = repo_index.get("call_graph", {}).get("reverse_neighbors", {})
    reachable = set(_reverse_reachable(reverse_neighbors, target_file, max_depth=6))
    matches = [item for item in primary_files if item == target_file or item in reachable]
    return matches[:8]


def resolve_modification_closure(
    repo_index: Dict[str, Any],
    target_file: str,
    target_symbols: Optional[List[str]] = None,
    planned_entrypoints: Optional[List[str]] = None,
    synchronized_edits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    target_file = _normalize_rel_path(target_file)
    manifest = repo_index.get("repo_manifest", {}).get("files", {})
    file_neighbors = repo_index.get("call_graph", {}).get("file_neighbors", {})
    reverse_neighbors = repo_index.get("call_graph", {}).get("reverse_neighbors", {})
    target_meta = manifest.get(target_file, {})
    upstream = _reverse_reachable(reverse_neighbors, target_file, max_depth=3)
    downstream = _forward_reachable(file_neighbors, target_file, max_depth=2)
    entrypoints = planned_entrypoints or _find_entrypoints_for_file(repo_index, target_file)
    sync_files = [_normalize_rel_path(item) for item in (synchronized_edits or []) if _normalize_rel_path(item)]
    config_files = [
        path for path, meta in manifest.items()
        if meta.get("is_config")
    ][:8]
    registry_files = [
        path for path, meta in manifest.items()
        if "registry" in meta.get("role_tags", []) or "factory" in meta.get("role_tags", [])
    ][:8]

    required_context = []
    for item in [target_file] + entrypoints + upstream[:6] + downstream[:6] + sync_files + registry_files[:4] + config_files[:4]:
        if item and item != target_file and item not in required_context:
            required_context.append(item)

    shared_interfaces = []
    if target_file in repo_index.get("symbol_index", {}).get("files", {}):
        target_symbols_index = repo_index["symbol_index"]["files"][target_file].get("symbols", [])
        for symbol in target_symbols_index[:8]:
            name = symbol.get("name", "")
            for owner in repo_index.get("symbol_index", {}).get("symbol_to_files", {}).get(name, []):
                if owner != target_file and owner not in shared_interfaces:
                    shared_interfaces.append(owner)

    optional_related = []
    target_dir = target_file.rsplit("/", 1)[0] if "/" in target_file else ""
    for path in sorted(manifest):
        if path == target_file or path in required_context:
            continue
        if target_dir and path.startswith(target_dir + "/"):
            optional_related.append(path)
    return {
        "path": target_file,
        "target_symbols": list(dict.fromkeys(target_symbols or [])),
        "entrypoints": list(dict.fromkeys(entrypoints)),
        "upstream_callers": upstream[:10],
        "downstream_callees": downstream[:10],
        "required_context_files": required_context[:16],
        "synchronized_edits": list(dict.fromkeys(sync_files))[:12],
        "shared_interface_files": shared_interfaces[:8],
        "config_and_registry_files": list(dict.fromkeys(registry_files[:4] + config_files[:4])),
        "optional_related_files": optional_related[:8],
        "focus_role_tags": target_meta.get("role_tags", []),
    }


def _normalize_modification_closure_item(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        path = _normalize_rel_path(item.get("path") or item.get("file") or item.get("target_file"))
        if not path:
            return None
        return {
            "path": path,
            "target_symbols": _extract_planned_symbols(item.get("target_symbols", [])),
            "upstream_callers": [str(x).strip() for x in item.get("upstream_callers", []) if str(x).strip()],
            "downstream_callees": [str(x).strip() for x in item.get("downstream_callees", []) if str(x).strip()],
            "required_context_files": [_normalize_rel_path(x) for x in item.get("required_context_files", []) if _normalize_rel_path(x)],
            "entrypoints": [str(x).strip() for x in item.get("entrypoints", []) if str(x).strip()],
            "synchronized_edits": [_normalize_rel_path(x) for x in item.get("synchronized_edits", []) if _normalize_rel_path(x)],
        }
    return None


def build_modification_closure(
    repo_index: Dict[str, Any],
    task_payload: Dict[str, Any],
    feature_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    feature_metadata = feature_metadata or {}
    task_list = task_payload.get("Task list", []) if isinstance(task_payload, dict) else []
    explicit_items = task_payload.get("Modification Closure", []) if isinstance(task_payload, dict) else []
    explicit_map: Dict[str, Dict[str, Any]] = {}
    for item in explicit_items:
        normalized = _normalize_modification_closure_item(item)
        if normalized:
            explicit_map[normalized["path"]] = normalized

    callsite_updates_by_file = feature_metadata.get("callsite_updates_by_file", {})
    public_changes_by_file = feature_metadata.get("public_interface_changes_by_file", {})
    replacement_targets = feature_metadata.get("core_replacement_targets", [])

    symbols_by_file: Dict[str, List[str]] = defaultdict(list)
    for row in replacement_targets:
        if len(row) < 2:
            continue
        file_key = _normalize_rel_path(row[0])
        symbol = str(row[1]).strip()
        if file_key and symbol:
            symbols_by_file[file_key].append(symbol)

    modification_closure_by_file: Dict[str, Any] = {}
    context_bundle_by_file: Dict[str, Any] = {}
    for item in task_list:
        path = _normalize_rel_path(item if isinstance(item, str) else item.get("file", ""))
        if not path:
            continue
        explicit = explicit_map.get(path, {})
        target_symbols = list(dict.fromkeys(symbols_by_file.get(path, []) + explicit.get("target_symbols", [])))
        sync_files = explicit.get("synchronized_edits", []) or []
        if not sync_files:
            sync_files = [
                _normalize_rel_path(candidate)
                for candidate in list(callsite_updates_by_file.get(path, [])) + list(public_changes_by_file.get(path, []))
                for candidate in re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9._-]+", candidate)
            ]
        closure = resolve_modification_closure(
            repo_index,
            path,
            target_symbols=target_symbols,
            planned_entrypoints=[_normalize_rel_path(x) for x in explicit.get("entrypoints", []) if _normalize_rel_path(x)],
            synchronized_edits=sync_files,
        )
        if explicit.get("required_context_files"):
            merged_required = closure["required_context_files"] + explicit["required_context_files"]
            closure["required_context_files"] = list(dict.fromkeys([x for x in merged_required if x and x != path]))[:16]
        closure["upstream_callers"] = list(dict.fromkeys(explicit.get("upstream_callers", []) + closure["upstream_callers"]))[:10]
        closure["downstream_callees"] = list(dict.fromkeys(explicit.get("downstream_callees", []) + closure["downstream_callees"]))[:10]
        interface_constraints = list(dict.fromkeys(public_changes_by_file.get(path, [])))
        closure["interface_constraints"] = interface_constraints[:12]
        modification_closure_by_file[path] = closure
        context_bundle_by_file[path] = {
            "focus_file": path,
            "required_context_files": closure["required_context_files"],
            "upstream_caller_files": [
                _extract_path_token(x)
                for x in closure["upstream_callers"]
                if _extract_path_token(x).endswith(PROMPT_RELEVANT_EXTENSIONS)
            ],
            "downstream_callee_files": [
                _extract_path_token(x)
                for x in closure["downstream_callees"]
                if _extract_path_token(x).endswith(PROMPT_RELEVANT_EXTENSIONS)
            ],
            "shared_interface_files": closure["shared_interface_files"],
            "config_and_registry_files": closure["config_and_registry_files"],
            "optional_related_files": closure["optional_related_files"],
            "entrypoint_chain": closure["entrypoints"],
            "synchronized_edit_targets": closure["synchronized_edits"],
            "target_symbols": closure["target_symbols"],
            "interface_constraints": closure["interface_constraints"],
        }
    return {
        "modification_closure": modification_closure_by_file,
        "context_bundle": context_bundle_by_file,
    }


def _load_file_from_repo(rel_path: str, primary_repo_dir: Optional[str], secondary_repo_dir: Optional[str] = None) -> Optional[Dict[str, str]]:
    rel_path = _normalize_rel_path(rel_path)
    for source_name, repo_dir in (("secondary", secondary_repo_dir), ("primary", primary_repo_dir)):
        if not repo_dir:
            continue
        abs_path = os.path.join(repo_dir, rel_path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            return {
                "path": rel_path,
                "source": source_name,
                "content": _safe_read_text(abs_path),
            }
    return None


def collect_context_bundle(
    repo_index: Dict[str, Any],
    closure: Dict[str, Any],
    primary_repo_dir: Optional[str],
    secondary_repo_dir: Optional[str] = None,
) -> Dict[str, Any]:
    focus_file = _normalize_rel_path(closure.get("path"))
    focus_payload = _load_file_from_repo(focus_file, primary_repo_dir, secondary_repo_dir)
    required_files = [_normalize_rel_path(x) for x in closure.get("required_context_files", []) if _normalize_rel_path(x)]
    upstream_files = [_extract_path_token(x) for x in closure.get("upstream_callers", []) if _extract_path_token(x)]
    downstream_files = [_extract_path_token(x) for x in closure.get("downstream_callees", []) if _extract_path_token(x)]
    shared_files = [_normalize_rel_path(x) for x in closure.get("shared_interface_files", []) if _normalize_rel_path(x)]
    cfg_registry = [_normalize_rel_path(x) for x in closure.get("config_and_registry_files", []) if _normalize_rel_path(x)]
    optional_files = [_normalize_rel_path(x) for x in closure.get("optional_related_files", []) if _normalize_rel_path(x)]

    def _materialize(paths: List[str], limit: int = 12) -> List[Dict[str, str]]:
        items = []
        for path in paths[:limit]:
            payload = _load_file_from_repo(path, primary_repo_dir, secondary_repo_dir)
            if payload:
                items.append(payload)
        return items

    return {
        "focus_file": focus_payload,
        "required_context_files": _materialize(required_files, limit=16),
        "upstream_callers_full": _materialize(upstream_files, limit=10),
        "downstream_callees_full": _materialize(downstream_files, limit=10),
        "shared_interfaces": _materialize(shared_files, limit=8),
        "config_and_registry_files": _materialize(cfg_registry, limit=8),
        "optional_related_files": _materialize(optional_files, limit=8),
        "entrypoint_chain": closure.get("entrypoints", []),
        "target_symbols": closure.get("target_symbols", []),
        "synchronized_edit_targets": closure.get("synchronized_edits", []),
        "interface_constraints": closure.get("interface_constraints", []),
        "focus_role_tags": closure.get("focus_role_tags", []),
        "repo_primary_entrypoints": repo_index.get("entrypoint_index", {}).get("primary_files", []),
    }


def _render_file_group(title: str, files: List[Dict[str, str]], truncate_each: Optional[int] = None) -> str:
    if not files:
        return f"## {title}\n(none)"
    sections = [f"## {title}"]
    for item in files:
        content = item.get("content", "")
        if truncate_each is not None and len(content) > truncate_each:
            content = content[:truncate_each] + "\n...(truncated for token budget)..."
        ext = os.path.splitext(item.get("path", ""))[1].lower()
        if ext == ".py":
            lang = "python"
        elif ext in (".yaml", ".yml"):
            lang = "yaml"
        elif ext == ".json":
            lang = "json"
        elif ext == ".sh":
            lang = "bash"
        else:
            lang = ""
        sections.append(
            f"### {item.get('path')} (source={item.get('source')})\n```{lang}\n{content}\n```"
        )
    return "\n".join(sections)


def render_context_bundle_for_prompt(context_bundle: Dict[str, Any]) -> Dict[str, str]:
    focus_payload = context_bundle.get("focus_file")
    focus_text = "(new file — no baseline/live code)"
    focus_source = "new"
    if focus_payload:
        focus_text = focus_payload.get("content", "")
        focus_source = focus_payload.get("source", "primary")
    return {
        "focus_file_code": focus_text,
        "focus_file_source": focus_source,
        "required_context_code": _render_file_group("Required Context Files", context_bundle.get("required_context_files", [])),
        "upstream_callers_code": _render_file_group("Upstream Callers", context_bundle.get("upstream_callers_full", [])),
        "downstream_callees_code": _render_file_group("Downstream Callees", context_bundle.get("downstream_callees_full", [])),
        "shared_interfaces_code": _render_file_group("Shared Interfaces", context_bundle.get("shared_interfaces", []), truncate_each=5000),
        "config_and_registry_code": _render_file_group("Config And Registry Files", context_bundle.get("config_and_registry_files", []), truncate_each=5000),
        "optional_related_code": _render_file_group("Optional Related Files", context_bundle.get("optional_related_files", []), truncate_each=3000),
        "entrypoint_chain": "\n".join(f"- {item}" for item in context_bundle.get("entrypoint_chain", [])) or "(none)",
        "synchronized_edit_targets": "\n".join(f"- {item}" for item in context_bundle.get("synchronized_edit_targets", [])) or "(none)",
        "interface_constraints": "\n".join(f"- {item}" for item in context_bundle.get("interface_constraints", [])) or "(none)",
        "target_symbols": "\n".join(f"- {item}" for item in context_bundle.get("target_symbols", [])) or "(none)",
        "repo_primary_entrypoints": "\n".join(f"- {item}" for item in context_bundle.get("repo_primary_entrypoints", [])) or "(none)",
    }
