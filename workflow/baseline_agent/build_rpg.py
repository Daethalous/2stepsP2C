"""
Build RPG — Stage 3.5 of the RPG-enhanced baseline pipeline.

Converts the flat planning output (architecture design, logic analysis,
todo_file_lst) into a structured RPG (Repository Planning Graph), then
topologically sorts files by their dependency relationships.

Usage (standalone test):
    python -m workflow.baseline_agent.build_rpg --output_dir <planning_output_dir>

This will:
  1. Load planning artifacts from the given directory
  2. Build an RPG graph with file nodes and dependency edges
  3. Print the dependency graph and topologically-sorted file order
  4. Save rpg_graph.json to the output directory
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any

from core.data_loader import load_pipeline_context, sanitize_todo_file_name
from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight RPG data structures (self-contained, no dependency on RPG-ZeroRepo)
# ---------------------------------------------------------------------------
# We define our own minimal graph here so this module works standalone.
# This mirrors the key concepts from zerorepo.rpg_gen.base.rpg but is
# tailored for the baseline pipeline's needs.

@dataclass
class FileNode:
    """A node representing a single file in the RPG."""
    path: str                       # e.g. "models/encoder.py"
    description: str = ""           # from logic_analysis
    directory: str = ""             # e.g. "models"
    features: List[str] = field(default_factory=list)  # extracted feature keywords
    depends_on: List[str] = field(default_factory=list)  # files this file imports from
    depended_by: List[str] = field(default_factory=list)  # files that import from this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineRPG:
    """
    A lightweight Repository Planning Graph for the baseline pipeline.

    Nodes = files (FileNode)
    Edges = dependency relationships (file A depends on file B)

    This is intentionally simpler than the full RPG from RPG-ZeroRepo.
    It focuses on the two things that matter most for code generation:
      1. File ordering via topological sort
      2. Dependency-aware context (which files to show when coding file X)
    """
    repo_name: str = ""
    nodes: Dict[str, FileNode] = field(default_factory=dict)  # path -> FileNode
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (src, dst) = src depends on dst
    subtrees: Dict[str, List[str]] = field(default_factory=dict)  # directory -> [file_paths]

    def add_file(self, path: str, description: str = "", features: List[str] = None):
        """Add a file node to the graph."""
        if path in self.nodes:
            # Update existing node
            node = self.nodes[path]
            if description:
                node.description = description
            if features:
                node.features.extend(features)
            return node

        directory = os.path.dirname(path) or "."
        node = FileNode(
            path=path,
            description=description,
            directory=directory,
            features=features or [],
        )
        self.nodes[path] = node

        # Track subtrees (directory groupings)
        if directory not in self.subtrees:
            self.subtrees[directory] = []
        self.subtrees[directory].append(path)

        return node

    def add_dependency(self, src: str, dst: str):
        """Add a dependency edge: src depends on dst (src imports from dst)."""
        if src not in self.nodes:
            logger.warning(f"Dependency source '{src}' not in graph, skipping")
            return
        if dst not in self.nodes:
            logger.warning(f"Dependency target '{dst}' not in graph, skipping")
            return
        if src == dst:
            return

        edge = (src, dst)
        if edge not in self.edges:
            self.edges.append(edge)
            self.nodes[src].depends_on.append(dst)
            self.nodes[dst].depended_by.append(src)

    def get_dependencies(self, file_path: str) -> List[str]:
        """Get all files that file_path directly depends on."""
        node = self.nodes.get(file_path)
        return list(node.depends_on) if node else []

    def get_dependents(self, file_path: str) -> List[str]:
        """Get all files that depend on file_path."""
        node = self.nodes.get(file_path)
        return list(node.depended_by) if node else []

    def get_transitive_dependencies(self, file_path: str) -> List[str]:
        """Get all files that file_path transitively depends on (BFS)."""
        visited = set()
        queue = deque(self.get_dependencies(file_path))
        while queue:
            dep = queue.popleft()
            if dep in visited:
                continue
            visited.add(dep)
            queue.extend(self.get_dependencies(dep))
        return list(visited)

    def get_same_subtree_files(self, file_path: str) -> List[str]:
        """Get all files in the same directory as file_path."""
        node = self.nodes.get(file_path)
        if not node:
            return []
        return [f for f in self.subtrees.get(node.directory, []) if f != file_path]

    def topological_sort(self) -> List[str]:
        """
        Topological sort of all file nodes using Kahn's algorithm.

        Files with no dependencies come first.
        If cycles exist, appends remaining nodes in alphabetical order.

        Returns:
            Ordered list of file paths.
        """
        # Build in-degree map
        in_degree = {path: 0 for path in self.nodes}
        for src, dst in self.edges:
            # src depends on dst → dst must come before src
            # So the edge in the DAG is dst → src
            in_degree[src] = in_degree.get(src, 0) + 1

        # Start with nodes that have no dependencies
        queue = deque(sorted(
            [path for path, deg in in_degree.items() if deg == 0]
        ))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # For each file that depends on this node, reduce its in-degree
            for dependent in sorted(self.get_dependents(node)):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Handle cycles: append remaining nodes
        if len(result) < len(self.nodes):
            remaining = sorted([p for p in self.nodes if p not in result])
            logger.warning(
                f"Cycle detected in dependency graph. "
                f"{len(remaining)} files in cycle: {remaining[:5]}..."
            )
            result.extend(remaining)

        return result

    def to_dict(self) -> dict:
        return {
            "repo_name": self.repo_name,
            "nodes": {path: node.to_dict() for path, node in self.nodes.items()},
            "edges": self.edges,
            "subtrees": self.subtrees,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineRPG":
        rpg = cls(repo_name=data.get("repo_name", ""))
        for path, node_dict in data.get("nodes", {}).items():
            rpg.nodes[path] = FileNode.from_dict(node_dict)
        rpg.edges = [tuple(e) for e in data.get("edges", [])]
        rpg.subtrees = data.get("subtrees", {})
        return rpg

    def save(self, path: str):
        """Save RPG to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"RPG saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PipelineRPG":
        """Load RPG from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# LLM-based dependency extraction (matches RPG-ZeroRepo's approach)
# ---------------------------------------------------------------------------

_DEP_GRAPH_SYSTEM_PROMPT = """\
You are an expert software architect. Given a list of files and their descriptions \
from a project plan, determine the dependency relationships between them.

A file A "depends on" file B means that A will import from or directly use \
classes/functions/data defined in B. The dependency direction is: A -> B \
(A depends on B, so B must be implemented before A).

Rules:
1. Only include dependencies between the files in the provided list.
2. The graph MUST be a DAG (Directed Acyclic Graph) - no circular dependencies.
3. Config files (e.g. config.yaml) have NO dependencies (they are read by others).
4. Non-code files (requirements.txt, docs.md, etc.) have NO dependencies.
5. Entry points (main.py, train.py, run.py) typically depend on most other .py files.
6. Test files (test_*.py, tests.py) depend on the modules they test.
7. Utility files (utils.py, helpers.py) are typically depended ON, not depending on others.
8. Be conservative: only add an edge if there is a clear import/usage relationship.

Return ONLY a JSON array where each element is:
{"name": "<file_path>", "depends_on": ["<dep1>", "<dep2>", ...]}

Include ALL files from the list, even those with empty depends_on.
"""

_DEP_GRAPH_USER_TEMPLATE = """\
## Project Files

{file_descriptions}

## Task
Determine the dependency graph for the above files.
Return ONLY a valid JSON array. No explanation, no markdown fences.
"""


def _extract_deps_via_llm(
    file_descriptions: Dict[str, str],
    all_files: List[str],
    gpt_version: str = "gpt-5-mini",
    max_retries: int = 2,
) -> Dict[str, List[str]]:
    """
    Ask an LLM to determine file dependencies from planning descriptions.

    This mirrors RPG-ZeroRepo's _plan_file_order_for_subtree() approach:
    the LLM sees the file list + descriptions and returns a dependency graph.

    Args:
        file_descriptions: file_path -> logic analysis description
        all_files: list of all file paths
        gpt_version: LLM model to use
        max_retries: number of retry attempts

    Returns:
        Dict mapping file_path -> list of dependencies
    """
    from core.llm_engine import create_client, chat_completion_with_retry

    # Build the file descriptions block
    desc_lines = []
    for file_path in all_files:
        desc = file_descriptions.get(file_path, "")
        summary = desc[:500] if desc else "(no description)"
        desc_lines.append(f"### {file_path}\n{summary}\n")

    user_content = _DEP_GRAPH_USER_TEMPLATE.format(
        file_descriptions="\n".join(desc_lines)
    )

    messages = [
        {"role": "system", "content": _DEP_GRAPH_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    client = create_client()
    file_set = set(all_files)
    response_text = ""

    for attempt in range(max_retries + 1):
        try:
            completion = chat_completion_with_retry(client, gpt_version, messages)
            response_text = completion.choices[0].message.content.strip()

            # Strip markdown fences if present
            if response_text.startswith("```"):
                response_text = re.sub(r"^```\w*\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            dep_list = json.loads(response_text)

            if not isinstance(dep_list, list):
                raise ValueError("Expected a JSON array")

            # Parse and validate
            dep_graph: Dict[str, List[str]] = {}
            errors = []

            for item in dep_list:
                name = item.get("name", "")
                depends_on = item.get("depends_on", [])

                if not isinstance(depends_on, list):
                    depends_on = []

                # Validate references
                valid_deps = []
                for dep in depends_on:
                    if dep in file_set and dep != name:
                        valid_deps.append(dep)
                    elif dep not in file_set:
                        errors.append(f"Invalid dep: {name} -> {dep} (not in file list)")

                dep_graph[name] = valid_deps

            # Check for cycles using DFS
            has_cycle, cycle_info = _detect_cycle(dep_graph)
            if has_cycle:
                logger.warning(f"  [LLM DEP] Cycle detected: {cycle_info}, retrying...")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": (
                    f"Your dependency graph has a cycle: {cycle_info}. "
                    "Please fix it. The graph MUST be a DAG. "
                    "Remove the weakest dependency to break the cycle. "
                    "Return the corrected JSON array."
                )})
                continue

            if errors:
                for e in errors[:5]:
                    logger.warning(f"  [LLM DEP] {e}")

            logger.info(f"  [LLM DEP] Successfully extracted {sum(len(v) for v in dep_graph.values())} dependencies")
            return dep_graph

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"  [LLM DEP] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": (
                    "Your output was not valid JSON. "
                    "Return ONLY a JSON array, no markdown, no explanation."
                )})

    logger.warning("  [LLM DEP] All attempts failed, returning empty dependencies")
    return {f: [] for f in all_files}


def _detect_cycle(dep_graph: Dict[str, List[str]]) -> Tuple[bool, str]:
    """Detect cycles in a dependency graph using DFS."""
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in dep_graph.get(node, []):
            if neighbor not in visited:
                result = dfs(neighbor)
                if result:
                    return result
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                return " -> ".join(cycle)

        rec_stack.discard(node)
        path.pop()
        return None

    for node in dep_graph:
        if node not in visited:
            result = dfs(node)
            if result:
                return True, result

    return False, ""


def _infer_structural_dependencies(file_path: str, all_files: List[str]) -> List[str]:
    """
    Infer structural dependencies based on file naming conventions.

    Rules:
      - __init__.py depends on all other .py files in same directory
      - test_*.py depends on the file it tests
    """
    deps = []
    basename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)

    if basename == "__init__.py":
        for other in all_files:
            if other == file_path:
                continue
            if os.path.dirname(other) == dirname and other.endswith(".py"):
                deps.append(other)

    elif basename.startswith("test_"):
        tested_name = basename.replace("test_", "", 1)
        for other in all_files:
            if os.path.basename(other) == tested_name:
                deps.append(other)

    return deps


# ---------------------------------------------------------------------------
# Main: Build RPG from planning output
# ---------------------------------------------------------------------------

def build_rpg_from_planning(
    output_dir: str,
    gpt_version: str = "gpt-5-mini",
    use_llm: bool = True,
) -> PipelineRPG:
    """
    Build a PipelineRPG from the planning artifacts in output_dir.

    Steps:
      1. Load planning context (todo_file_lst, logic_analysis_dict)
      2. Create file nodes with descriptions
      3. Extract dependencies via LLM (or structural heuristics as fallback)
      4. Topological sort

    Args:
        output_dir: Directory containing planning_trajectories.json,
                    planning_config.yaml, etc.
        gpt_version: LLM model for dependency extraction.
        use_llm: If True, use LLM for dependency extraction.
                 If False, use only structural heuristics.

    Returns:
        PipelineRPG with nodes, edges, and sorted file order.
    """
    logger.info("=== Building RPG from planning output ===")

    # 1. Load planning context
    ctx = load_pipeline_context(output_dir)
    todo_file_lst = ctx.todo_file_lst
    logic_analysis_dict = dict(ctx.logic_analysis_dict)

    logger.info(f"  Files in todo_file_lst: {len(todo_file_lst)}")
    logger.info(f"  Files with logic analysis: {len(logic_analysis_dict)}")

    # 2. Create RPG and add file nodes
    rpg = PipelineRPG(repo_name=os.path.basename(output_dir))

    for file_path in todo_file_lst:
        desc = logic_analysis_dict.get(file_path, "")
        rpg.add_file(path=file_path, description=desc)

    logger.info(f"  Created {len(rpg.nodes)} file nodes")
    logger.info(f"  Directory subtrees: {list(rpg.subtrees.keys())}")

    # 3. Extract dependencies
    llm_dep_count = 0
    if use_llm:
        logger.info("  Extracting dependencies via LLM...")
        dep_graph = _extract_deps_via_llm(
            file_descriptions=logic_analysis_dict,
            all_files=todo_file_lst,
            gpt_version=gpt_version,
        )

        for file_path, deps in dep_graph.items():
            for dep in deps:
                if dep != file_path and dep in rpg.nodes and file_path in rpg.nodes:
                    rpg.add_dependency(file_path, dep)
                    llm_dep_count += 1

        logger.info(f"  LLM dependencies added: {llm_dep_count}")

    # 4. Add structural dependency heuristics (supplementary only)
    struct_dep_count = 0
    for file_path in todo_file_lst:
        struct_deps = _infer_structural_dependencies(file_path, todo_file_lst)
        for dep in struct_deps:
            if (file_path, dep) not in rpg.edges:
                rpg.add_dependency(file_path, dep)
                struct_dep_count += 1

    logger.info(f"  Structural dependencies: {struct_dep_count}")
    logger.info(f"  Total edges: {len(rpg.edges)}")

    # 5. Topological sort
    sorted_files = rpg.topological_sort()
    logger.info(f"  Topologically sorted file order computed")

    return rpg


def compare_file_orders(original: List[str], rpg_sorted: List[str]) -> str:
    """
    Generate a comparison report between original and RPG-sorted file orders.
    Useful for verifying the RPG is improving dependency ordering.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("FILE ORDER COMPARISON: Original vs RPG-Sorted")
    lines.append("=" * 70)
    lines.append("")

    max_len = max(len(original), len(rpg_sorted))
    lines.append(f"{'#':<4} {'Original':<45} {'RPG-Sorted':<45}")
    lines.append("-" * 94)

    for i in range(max_len):
        orig = original[i] if i < len(original) else "(n/a)"
        rpg = rpg_sorted[i] if i < len(rpg_sorted) else "(n/a)"
        marker = " *" if orig != rpg else ""
        lines.append(f"{i+1:<4} {orig:<45} {rpg:<45}{marker}")

    lines.append("")

    # Count position changes
    changes = 0
    for i, f in enumerate(rpg_sorted):
        if i < len(original) and original[i] != f:
            changes += 1

    lines.append(f"Files reordered: {changes} / {len(original)}")

    # Check for files that moved significantly
    orig_pos = {f: i for i, f in enumerate(original)}
    big_moves = []
    for i, f in enumerate(rpg_sorted):
        if f in orig_pos:
            delta = orig_pos[f] - i
            if abs(delta) >= 3:
                direction = "earlier" if delta > 0 else "later"
                big_moves.append(f"  {f}: moved {abs(delta)} positions {direction}")

    if big_moves:
        lines.append("")
        lines.append("Significant position changes (>=3 positions):")
        lines.extend(big_moves)

    return "\n".join(lines)


def print_dependency_graph(rpg: PipelineRPG):
    """Print a human-readable view of the dependency graph."""
    lines = []
    lines.append("=" * 70)
    lines.append("DEPENDENCY GRAPH")
    lines.append("=" * 70)

    for path in sorted(rpg.nodes.keys()):
        node = rpg.nodes[path]
        lines.append(f"\n  {path}")
        if node.depends_on:
            for dep in sorted(node.depends_on):
                lines.append(f"    <- depends on: {dep}")
        if node.depended_by:
            for dep in sorted(node.depended_by):
                lines.append(f"    -> depended by: {dep}")
        if not node.depends_on and not node.depended_by:
            lines.append(f"    (no dependencies)")

    lines.append("")
    lines.append(f"Total nodes: {len(rpg.nodes)}")
    lines.append(f"Total edges: {len(rpg.edges)}")
    lines.append(f"Subtrees: {list(rpg.subtrees.keys())}")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build RPG from planning output (Stage 3.5)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory containing planning artifacts (planning_trajectories.json, etc.)"
    )
    parser.add_argument(
        "--gpt_version", type=str, default="gpt-5-mini",
        help="LLM model for dependency extraction (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--no_llm", action="store_true", default=False,
        help="Skip LLM dependency extraction, use only structural heuristics"
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save RPG to rpg_graph.json in output_dir"
    )
    args = parser.parse_args()

    # Build RPG
    rpg = build_rpg_from_planning(
        args.output_dir,
        gpt_version=args.gpt_version,
        use_llm=not args.no_llm,
    )

    # Print dependency graph
    print_dependency_graph(rpg)

    # Get topologically sorted order
    sorted_files = rpg.topological_sort()

    # Load original order for comparison
    ctx = load_pipeline_context(args.output_dir)
    original_order = ctx.todo_file_lst

    # Print comparison
    comparison = compare_file_orders(original_order, sorted_files)
    print("\n" + comparison)

    # Save RPG
    if args.save:
        rpg_path = os.path.join(args.output_dir, "rpg_graph.json")
        rpg.save(rpg_path)

        # Also save the sorted file list
        sorted_path = os.path.join(args.output_dir, "rpg_sorted_files.json")
        with open(sorted_path, "w", encoding="utf-8") as f:
            json.dump({
                "original_order": original_order,
                "rpg_sorted_order": sorted_files,
            }, f, indent=2)
        logger.info(f"Sorted file orders saved to {sorted_path}")


if __name__ == "__main__":
    main()
