import os
from string import Template

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {}


def load_prompt(name: str) -> str:
    if name not in _cache:
        path = os.path.join(_PROMPTS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]


def render_prompt(name: str, **kwargs) -> str:
    template = Template(load_prompt(name))
    return template.safe_substitute(**kwargs)
