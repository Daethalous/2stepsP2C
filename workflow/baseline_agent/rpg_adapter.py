"""
RPG Adapter — graph query helpers for the baseline pipeline.

Provides high-level functions that downstream stages (analyzing, coding)
use to query the RPG for context-building:

  - get_coding_context():  Build focused context for coding a specific file
  - get_analysis_context(): Build upstream dependency context for analyzing
  - get_cross_module_interfaces(): Summarize cross-module APIs for api_predefine
  - get_stub_context(): Load verified interface stubs for coding enforcement
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger
from core.utils import extract_interface_signatures

logger = get_logger(__name__)


# Import PipelineRPG from our own module
from workflow.baseline_agent.build_rpg import PipelineRPG


def get_coding_context(
    rpg: PipelineRPG,
    target_file: str,
    done_file_dict: Dict[str, str],
    max_total_chars: int = 14000,
) -> str:
    """
    Build focused coding context for a target file using RPG dependencies.

    Instead of dumping ALL previously generated files, this provides:
      1. Full code for direct dependencies (files target imports from)
      2. Interface signatures for sibling files (same directory)
      3. Interface signatures for transitive dependencies (if budget allows)

    Args:
        rpg: The PipelineRPG graph.
        target_file: File being coded right now.
        done_file_dict: Map of file_path → generated code for completed files.
        max_total_chars: Token budget in characters.

    Returns:
        Formatted context string for the coding prompt.
    """
    parts = []
    total_chars = 0

    # --- Priority 1: Direct dependencies (full code) ---
    direct_deps = rpg.get_dependencies(target_file)
    for dep in direct_deps:
        if dep not in done_file_dict:
            continue
        code = done_file_dict[dep]
        chunk = f"\n### {dep} (direct dependency — full code)\n```python\n{code}\n```\n"
        if total_chars + len(chunk) > max_total_chars:
            # Fall back to signatures if full code is too large
            sig = extract_interface_signatures(code, max_lines=80)
            chunk = f"\n### {dep} (direct dependency — signatures)\n```python\n{sig}\n```\n"
            if total_chars + len(chunk) > max_total_chars:
                break
        parts.append(chunk)
        total_chars += len(chunk)

    # --- Priority 2: Sibling files (interface signatures) ---
    siblings = rpg.get_same_subtree_files(target_file)
    for sib in siblings:
        if sib not in done_file_dict or sib in direct_deps:
            continue
        sig = extract_interface_signatures(done_file_dict[sib], max_lines=40)
        chunk = f"\n### {sib} (sibling — signatures)\n```python\n{sig}\n```\n"
        if total_chars + len(chunk) > max_total_chars:
            break
        parts.append(chunk)
        total_chars += len(chunk)

    # --- Priority 3: Transitive dependencies (interface signatures) ---
    transitive_deps = rpg.get_transitive_dependencies(target_file)
    # Exclude direct deps and siblings already included
    already_included = set(direct_deps) | set(siblings)
    for dep in transitive_deps:
        if dep in already_included or dep not in done_file_dict:
            continue
        sig = extract_interface_signatures(done_file_dict[dep], max_lines=30)
        chunk = f"\n### {dep} (transitive dependency — signatures)\n```python\n{sig}\n```\n"
        if total_chars + len(chunk) > max_total_chars:
            parts.append("\n...(remaining dependency signatures truncated)...\n")
            break
        parts.append(chunk)
        total_chars += len(chunk)

    if not parts:
        return "(no dependency context available)"

    header = (
        f"## Dependency Context for {target_file}\n"
        f"Direct dependencies: {direct_deps or '(none)'}\n"
        f"Sibling files: {siblings or '(none)'}\n"
    )
    return header + "".join(parts)


def get_analysis_context(
    rpg: PipelineRPG,
    target_file: str,
    logic_analysis_dict: Dict[str, str],
    max_chars: int = 4000,
) -> str:
    """
    Build upstream dependency context for the analysis stage.

    For each file that target_file depends on, include its logic analysis
    so the LLM understands what APIs are available.

    Args:
        rpg: The PipelineRPG graph.
        target_file: File being analyzed.
        logic_analysis_dict: Map of file_path → logic analysis text.
        max_chars: Maximum characters.

    Returns:
        Formatted context string.
    """
    deps = rpg.get_dependencies(target_file)
    if not deps:
        return ""

    parts = [f"## Upstream Dependencies for {target_file}\n"]
    total = len(parts[0])

    for dep in deps:
        analysis = logic_analysis_dict.get(dep, "")
        if not analysis:
            continue
        # Truncate individual analyses
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


def get_cross_module_interfaces(
    rpg: PipelineRPG,
) -> str:
    """
    Summarize which files export/import across directory boundaries.

    Useful for the api_predefine stage to focus on cross-module APIs.

    Returns:
        Formatted summary of cross-module dependencies.
    """
    cross_module = []

    for src, dst in rpg.edges:
        src_dir = os.path.dirname(src) or "."
        dst_dir = os.path.dirname(dst) or "."
        if src_dir != dst_dir:
            cross_module.append((src, dst))

    if not cross_module:
        return "(no cross-module dependencies detected)"

    lines = ["## Cross-Module Dependencies\n"]

    # Group by source directory
    by_src_dir: Dict[str, List[Tuple[str, str]]] = {}
    for src, dst in cross_module:
        src_dir = os.path.dirname(src) or "."
        if src_dir not in by_src_dir:
            by_src_dir[src_dir] = []
        by_src_dir[src_dir].append((src, dst))

    for src_dir in sorted(by_src_dir.keys()):
        lines.append(f"\n### {src_dir}/")
        for src, dst in sorted(by_src_dir[src_dir]):
            lines.append(f"  {os.path.basename(src)} <- imports from {dst}")

    return "\n".join(lines)


def get_file_generation_order(rpg: PipelineRPG) -> List[str]:
    """Get the topologically sorted file order from the RPG."""
    return rpg.topological_sort()


# ---------------------------------------------------------------------------
# Interface stub helpers
# ---------------------------------------------------------------------------

def load_stubs_dict(output_dir: str) -> Dict[str, str]:
    """
    Load all verified stubs from the stubs/ directory.

    Returns:
        Dict mapping file_name -> stub code
    """
    stubs_dir = os.path.join(output_dir, "stubs")
    stubs_dict: Dict[str, str] = {}

    if not os.path.isdir(stubs_dir):
        logger.info("  [RPG] No stubs/ directory found — skipping stub loading")
        return stubs_dict

    for fname in os.listdir(stubs_dir):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(stubs_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        # Convert filename back: src_metrics.py -> src/metrics
        key = fname[:-3].replace("_", "/", 1) if "_" in fname else fname[:-3]
        # Try multiple key formats to match
        stubs_dict[key] = code
        # Also store with .py extension variant
        stubs_dict[key + ".py"] = code

    logger.info(f"  [RPG] Loaded {len(stubs_dict) // 2} stubs from {stubs_dir}")
    return stubs_dict


def get_stub_context(
    stubs_dict: Dict[str, str],
    target_file: str,
    rpg: PipelineRPG,
) -> Tuple[str, str]:
    """
    Build stub context for the coding stage.

    Returns:
        Tuple of:
          - own_stub: The stub for target_file itself (the contract it must implement)
          - dep_stubs: Formatted text showing stubs of all dependencies

    The coding prompt should include both:
      - own_stub: "You MUST implement exactly these signatures"
      - dep_stubs: "These are your dependencies — use them, don't reimplement"
    """
    # Own stub
    own_stub = stubs_dict.get(target_file, "")
    if not own_stub:
        # Try without extension
        base = target_file.replace(".py", "") if target_file.endswith(".py") else target_file
        own_stub = stubs_dict.get(base, "")

    # Dependency stubs
    deps = rpg.get_dependencies(target_file)
    dep_parts = []
    for dep in deps:
        stub = stubs_dict.get(dep, "")
        if not stub:
            base = dep.replace(".py", "") if dep.endswith(".py") else dep
            stub = stubs_dict.get(base, "")
        if stub:
            dep_parts.append(
                f"### {dep} (VERIFIED interface — use these exact signatures)\n"
                f"```python\n{stub}\n```\n"
            )

    dep_stubs = "\n".join(dep_parts) if dep_parts else ""

    return own_stub, dep_stubs

