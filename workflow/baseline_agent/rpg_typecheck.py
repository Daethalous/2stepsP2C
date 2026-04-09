"""
RPG Type Checking — Optional pyright integration for post-coding validation.

Runs pyright (if available) on the generated repository to catch:
  - Type mismatches across files
  - Missing imports / undefined names
  - Parameter type errors

If pyright is not installed, gracefully skips with a warning.

Usage:
    Called from rpg_pipeline.py as the "typecheck" stage.
    Can also run standalone:
        python -m workflow.baseline_agent.rpg_typecheck \\
            --repo_dir outputs/iTransformer_repo_rpg2
"""

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# pyrightconfig.json template
# ---------------------------------------------------------------------------

def _generate_pyright_config(repo_dir: str) -> str:
    """Generate a minimal pyrightconfig.json in the repo directory."""
    config = {
        "include": ["."],
        "exclude": [
            "**/node_modules",
            "**/__pycache__",
            "**/.git",
        ],
        "reportMissingImports": "warning",
        "reportMissingModuleSource": "none",
        "reportOptionalMemberAccess": "none",
        "reportGeneralTypeIssues": "warning",
        "reportAttributeAccessIssue": "none",
        "pythonVersion": "3.10",
        "typeCheckingMode": "basic",
    }
    config_path = os.path.join(repo_dir, "pyrightconfig.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


# ---------------------------------------------------------------------------
# Pyright output parsing
# ---------------------------------------------------------------------------

def _parse_pyright_output(output_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse pyright JSON output into a list of structured errors.

    Returns list of dicts:
        {
            "file": str,
            "line": int,
            "col": int,
            "severity": str,  # "error", "warning", "information"
            "message": str,
            "rule": str,
        }
    """
    errors = []
    diagnostics = output_json.get("generalDiagnostics", [])
    for diag in diagnostics:
        errors.append({
            "file": diag.get("file", "unknown"),
            "line": diag.get("range", {}).get("start", {}).get("line", 0),
            "col": diag.get("range", {}).get("start", {}).get("character", 0),
            "severity": diag.get("severity", "unknown"),
            "message": diag.get("message", ""),
            "rule": diag.get("rule", ""),
        })
    return errors


def _group_errors_by_file(
    errors: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group errors by file path."""
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for err in errors:
        fname = err["file"]
        # Normalize to relative path
        fname = os.path.basename(fname)
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(err)
    return by_file


def _format_errors_for_report(
    errors: List[Dict[str, Any]],
    repo_dir: str,
) -> str:
    """Format errors into a human-readable report."""
    if not errors:
        return "No type errors found!"

    by_file = _group_errors_by_file(errors)

    lines = [
        "Pyright Type Check Report",
        "=" * 50,
        f"Total issues: {len(errors)}",
        f"Files affected: {len(by_file)}",
        "",
    ]

    # Count by severity
    error_count = sum(1 for e in errors if e["severity"] == "error")
    warning_count = sum(1 for e in errors if e["severity"] == "warning")
    info_count = len(errors) - error_count - warning_count

    lines.append(f"  Errors: {error_count}")
    lines.append(f"  Warnings: {warning_count}")
    lines.append(f"  Info: {info_count}")
    lines.append("")

    for fname, file_errors in sorted(by_file.items()):
        lines.append(f"--- {fname} ({len(file_errors)} issues) ---")
        for err in file_errors:
            severity = err["severity"].upper()
            line_num = err["line"] + 1  # pyright uses 0-indexed
            msg = err["message"]
            rule = err["rule"]
            rule_str = f" [{rule}]" if rule else ""
            lines.append(f"  L{line_num}: [{severity}]{rule_str} {msg}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_typecheck(
    repo_dir: str,
    output_dir: str = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Run pyright type checking on the generated repository.

    Args:
        repo_dir: Path to the generated repository.
        output_dir: Where to save the report (optional).

    Returns:
        Tuple of:
          - success: bool (True if pyright ran, even with errors)
          - report: str (human-readable report)
          - errors: List of error dicts
    """
    logger.info("=== Type Check Stage (Pyright) ===")

    # Check if pyright is available
    pyright_path = shutil.which("pyright")
    if pyright_path is None:
        msg = (
            "  [SKIP] pyright not found on PATH. "
            "Install with: pip install pyright  OR  npm install -g pyright"
        )
        logger.warning(msg)
        return False, msg, []

    logger.info(f"  [OK] pyright found: {pyright_path}")

    # Generate config
    config_path = _generate_pyright_config(repo_dir)
    logger.info(f"  Generated pyrightconfig.json: {config_path}")

    # Run pyright
    try:
        result = subprocess.run(
            [pyright_path, "--outputjson"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
    except subprocess.TimeoutExpired:
        msg = "  [TIMEOUT] pyright timed out after 120 seconds."
        logger.error(msg)
        return False, msg, []
    except FileNotFoundError:
        msg = "  [ERROR] pyright executable not found."
        logger.error(msg)
        return False, msg, []

    # Parse output
    try:
        output_json = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        # pyright may output non-JSON on stderr
        msg = f"  [ERROR] Could not parse pyright output.\nstdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        logger.error(msg)
        return False, msg, []

    errors = _parse_pyright_output(output_json)

    # Build report
    report = _format_errors_for_report(errors, repo_dir)

    # Log summary
    error_count = sum(1 for e in errors if e["severity"] == "error")
    warning_count = sum(1 for e in errors if e["severity"] == "warning")
    logger.info(f"  Results: {error_count} errors, {warning_count} warnings")

    # Save report
    if output_dir:
        report_path = os.path.join(output_dir, "pyright_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"  Report saved: {report_path}")

        # Save raw JSON
        raw_path = os.path.join(output_dir, "pyright_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2)

    # Clean up generated config
    try:
        os.remove(config_path)
    except OSError:
        pass

    logger.info("=== Type Check Stage Complete ===")
    return True, report, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from core.logger import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="RPG Pyright Type Checker")
    parser.add_argument("--repo_dir", type=str, required=True,
                        help="Path to the generated repository")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the report")
    args = parser.parse_args()

    success, report, errors = run_typecheck(
        repo_dir=args.repo_dir,
        output_dir=args.output_dir or args.repo_dir,
    )
    print(report)
