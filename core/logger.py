import logging
import sys


def setup_logging(level=logging.INFO):
    root = logging.getLogger("papercoder")
    if root.handlers:
        return root
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"papercoder.{name}")
