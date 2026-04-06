from __future__ import annotations

import os
from datetime import datetime


def make_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_run_tag(path: str, run_tag: str) -> str:
    if not run_tag:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}_{run_tag}{ext}"
    return f"{path}_{run_tag}"


def resolve_default_output_path(path: str, default_path: str, run_tag: str) -> str:
    if os.path.normpath(path) != os.path.normpath(default_path):
        return path
    return append_run_tag(default_path, run_tag)
