import json
import re
import os
import yaml
from datetime import datetime
from core.logger import get_logger

logger = get_logger(__name__)


def extract_planning(trajectories_json_file_path):
    with open(trajectories_json_file_path) as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            content = turn['content']
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)

    context_lst = context_lst[:3]

    return context_lst


def _apply_cleaners(data, strip_comments_after_comma=False,
                    strip_comments_after_string=False,
                    fix_triple_quotes=False):
    clean = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    if strip_comments_after_comma:
        clean = re.sub(r'(".*?"),\s*#.*', r'\1,', clean)
    if strip_comments_after_string:
        clean = re.sub(r'(".*?")\s*#.*', r'\1', clean)
    clean = re.sub(r',\s*\]', ']', clean)
    clean = re.sub(r'\n\s*', '', clean)
    if fix_triple_quotes:
        clean = re.sub(r'"""', '"', clean)
        clean = re.sub(r"'''", "'", clean)
        clean = re.sub(r"\\", "'", clean)
    return json.loads(clean)


def _extract_logic_and_task_fallback(data):
    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)
    if match:
        return {
            "Logic Analysis": json.loads(match.group(1)),
            "Task list": json.loads(match.group(2)),
        }
    return {}


_CONTENT_TO_JSON_CLEANERS = [
    lambda d: _apply_cleaners(d, strip_comments_after_comma=True),
    lambda d: _apply_cleaners(d, strip_comments_after_comma=True,
                              strip_comments_after_string=True),
    lambda d: _apply_cleaners(d, strip_comments_after_comma=True,
                              strip_comments_after_string=True,
                              fix_triple_quotes=True),
    _extract_logic_and_task_fallback,
]


def content_to_json(data):
    for cleaner in _CONTENT_TO_JSON_CLEANERS:
        try:
            result = cleaner(data)
            if result is not None:
                return result
        except (json.JSONDecodeError, Exception):
            continue
    return {}


def extract_content_block(text: str) -> str:
    """Extract payload inside [CONTENT]...[/CONTENT]."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"\[CONTENT\](.*?)\[/CONTENT\]", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_structured_json(text: str) -> dict:
    """Parse structured JSON payload from model output."""
    payload = extract_content_block(text)
    try:
        return json.loads(payload)
    except Exception:
        return content_to_json(payload)


def validate_required_keys(data: dict, required_keys: list) -> bool:
    if not isinstance(data, dict):
        return False
    for key in required_keys:
        if key not in data:
            return False
    return True


def contains_forbidden_placeholders(text: str) -> bool:
    if not isinstance(text, str):
        return False
    low = text.lower()
    markers = ["notimplementederror", "# todo", "todo:", "placeholder", "stub implementation"]
    return any(m in low for m in markers)


def extract_code_from_content(content):
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if len(code) == 0:
        return ""
    else:
        return code[0]


def format_json_data(data):
    formatted_text = ""
    for key, value in data.items():
        formatted_text += "-" * 40 + "\n"
        formatted_text += "[" + key + "]\n"
        if isinstance(value, list):
            for item in value:
                formatted_text += f"- {item}\n"
        else:
            formatted_text += str(value) + "\n"
        formatted_text += "\n"
    return formatted_text


_model_cost_cache = None


def _load_model_cost():
    global _model_cost_cache
    if _model_cost_cache is None:
        pricing_path = os.path.join(os.path.dirname(__file__), "model_pricing.json")
        with open(pricing_path, "r", encoding="utf-8") as f:
            _model_cost_cache = json.load(f)
    return _model_cost_cache


def cal_cost(response_json, model_name):
    model_cost = _load_model_cost()

    prompt_tokens = response_json["usage"]["prompt_tokens"]
    completion_tokens = response_json["usage"]["completion_tokens"]
    cached_tokens = response_json["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

    actual_input_tokens = prompt_tokens - cached_tokens
    output_tokens = completion_tokens

    cost_info = model_cost[model_name]

    input_cost = (actual_input_tokens / 1_000_000) * cost_info['input']
    cached_input_cost = (cached_tokens / 1_000_000) * cost_info['cached_input'] if cost_info['cached_input'] else 0
    output_cost = (output_tokens / 1_000_000) * cost_info['output'] if cost_info['output'] else 0

    total_cost = input_cost + cached_input_cost + output_cost

    return {
        'model_name': model_name,
        'actual_input_tokens': actual_input_tokens,
        'input_cost': input_cost,
        'cached_tokens': cached_tokens,
        'cached_input_cost': cached_input_cost,
        'output_tokens': output_tokens,
        'output_cost': output_cost,
        'total_cost': total_cost,
    }


def load_accumulated_cost(accumulated_cost_file):
    if os.path.exists(accumulated_cost_file):
        with open(accumulated_cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("total_cost", 0.0)
    else:
        return 0.0


def save_accumulated_cost(accumulated_cost_file, cost):
    with open(accumulated_cost_file, "w", encoding="utf-8") as f:
        json.dump({"total_cost": cost}, f)


def print_response(completion_json, is_llm=False):
    logger.info("============================================")
    if is_llm:
        logger.info(completion_json['text'])
    else:
        logger.info(completion_json['choices'][0]['message']['content'])
    logger.info("============================================\n")


def print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost):
    usage_info = cal_cost(completion_json, gpt_version)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("\U0001F31F Usage Summary \U0001F31F")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"\U0001F6E0\uFE0F Model: {usage_info['model_name']}")
    output_lines.append(f"\U0001F4E5 Input tokens: {usage_info['actual_input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"\U0001F4E6 Cached input tokens: {usage_info['cached_tokens']} (Cost: ${usage_info['cached_input_cost']:.8f})")
    output_lines.append(f"\U0001F4E4 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"\U0001F4B5 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"\U0001FA99 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)

    logger.info(output_text)

    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")

    return total_accumulated_cost


def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    """Return the number of tokens used by a list of messages."""
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        logger.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        logger.warning("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        logger.warning("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, allowed_special={"<|endoftext|>"}, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def read_all_files(directory, allowed_ext, is_print=True):
    """Recursively read all files with allowed extensions and return their contents."""
    all_files_content = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            relative_path = os.path.relpath(os.path.join(root, filename), directory)

            _file_name, ext = os.path.splitext(filename)

            is_skip = False
            if len(directory) < len(root):
                root2 = root[len(directory)+1:]
                for dirname in root2.split("/"):
                    if dirname.startswith("."):
                        is_skip = True
                        break

            if filename.startswith(".") or "requirements.txt" in filename or ext == "" or is_skip:
                if is_print and ext == "":
                    logger.info(f"[SKIP] {os.path.join(root, filename)}")
                continue

            if ext not in allowed_ext:
                if _file_name.lower() != "readme":
                    if is_print:
                        logger.info(f"[SKIP] {os.path.join(root, filename)}")
                    continue

            try:
                filepath = os.path.join(root, filename)
                file_size = os.path.getsize(filepath)

                if file_size > 204800:
                    logger.warning(f"[BIG] {filepath} {file_size}")

                with open(filepath, "r") as file:
                    all_files_content[relative_path] = file.read()
            except Exception as e:
                logger.warning(e)
                logger.info(f"[SKIP] {os.path.join(root, filename)}")

    return all_files_content


def read_python_files(directory):
    """Recursively read all .py files in the specified directory and return their contents."""
    python_files_content = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                relative_path = os.path.relpath(os.path.join(root, filename), directory)
                with open(os.path.join(root, filename), "r", encoding="utf-8") as file:
                    python_files_content[relative_path] = file.read()

    return python_files_content


def extract_json_from_string(text):
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)

    logger.warning("No JSON content found.")
    return ""


def get_now_str():
    now = datetime.now()
    now = str(now)
    now = now.split(".")[0]
    now = now.replace("-", "").replace(" ", "_").replace(":", "")
    return now


def build_baseline_repo_context(baseline_repo_dir: str):
    """Read all .py and .yaml files from a baseline repo and return (summary, full_code).

    summary: concise list of filenames with first-line description.
    full_code: concatenated file contents for injection into prompts.
    """
    all_files = read_all_files(
        baseline_repo_dir,
        allowed_ext=[".py", ".yaml", ".yml"],
        is_print=False,
    )
    summary_lines = []
    code_parts = []
    for rel_path in sorted(all_files):
        content = all_files[rel_path]
        first_line = content.split("\n")[0].strip() if content.strip() else "(empty)"
        summary_lines.append(f"- {rel_path}: {first_line}")
        code_parts.append(f"### {rel_path}\n```\n{content}\n```")
    summary = "\n".join(summary_lines) if summary_lines else "(no files)"
    full_code = "\n\n".join(code_parts) if code_parts else "(no files)"
    return summary, full_code


def format_paper_content_for_prompt(paper_content, max_chars: int = 20000) -> str:
    """Convert paper content into a bounded string for prompt safety."""
    if isinstance(paper_content, (dict, list)):
        text = json.dumps(paper_content, ensure_ascii=False)
    else:
        text = str(paper_content)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated for token budget)..."


def extract_interface_signatures(code: str, max_lines: int = 120) -> str:
    """Extract lightweight interface signatures from source code."""
    sig_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("class ")
            or stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            sig_lines.append(line)
            if len(sig_lines) >= max_lines:
                break
    if not sig_lines:
        return "(no signatures found)"
    return "\n".join(sig_lines)


def build_baseline_repo_context_compact(
    baseline_repo_dir: str,
    max_files: int = 120,
    max_chars_per_file: int = 1600,
):
    """Build compact baseline context with summary + signature-focused code snippets."""
    all_files = read_all_files(
        baseline_repo_dir,
        allowed_ext=[".py", ".yaml", ".yml"],
        is_print=False,
    )
    summary_lines = []
    code_parts = []
    for rel_path in sorted(all_files)[:max_files]:
        content = all_files[rel_path]
        first_line = content.split("\n")[0].strip() if content.strip() else "(empty)"
        summary_lines.append(f"- {rel_path}: {first_line}")
        if rel_path.endswith(".py"):
            compact = extract_interface_signatures(content, max_lines=120)
        else:
            compact = content[:max_chars_per_file]
        if len(compact) > max_chars_per_file:
            compact = compact[:max_chars_per_file] + "\n...(truncated)..."
        code_parts.append(f"### {rel_path}\n```\n{compact}\n```")
    summary = "\n".join(summary_lines) if summary_lines else "(no files)"
    full_code = "\n\n".join(code_parts) if code_parts else "(no files)"
    return summary, full_code


def build_code_interface_summary(done_file_dict: dict, done_file_lst: list,
                                 max_total_chars: int = 12000) -> str:
    """Build compact interface summary for previously completed files."""
    parts = []
    total = 0
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"):
            continue
        code = done_file_dict.get(done_file, "")
        sig = extract_interface_signatures(code, max_lines=80)
        chunk = f"\n### {done_file}\n```python\n{sig}\n```\n"
        if total + len(chunk) > max_total_chars:
            parts.append("\n...(truncated interface summaries)...")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts) if parts else "(no previous code interfaces)"


def parse_feature_design(design_text: str):
    """Parse the feature planning design output to extract injection points and unchanged files."""
    design_json = content_to_json(design_text)
    if not design_json:
        logger.warning("Failed to parse feature design output. Using empty defaults.")
    injection_points = design_json.get("Injection points", [])
    injection_points = [p for p in injection_points if isinstance(p, dict) and "file" in p]
    unchanged_files = design_json.get("Files unchanged", [])
    if not isinstance(unchanged_files, list):
        unchanged_files = []
    new_files = design_json.get("New files needed", [])
    if not isinstance(new_files, list):
        new_files = []
    return injection_points, unchanged_files, new_files


def get_injection_info_for_file(injection_points: list, file_name: str) -> str:
    """Extract injection info string for a specific file from the injection points list."""
    infos = []
    for point in injection_points:
        if point.get("file", "") == file_name:
            action = point.get("action", "modify")
            location = point.get("location", "")
            description = point.get("description", "")
            infos.append(f"- Action: {action}, Location: {location}, Description: {description}")
    if infos:
        return "\n".join(infos)
    return f"No specific injection info found for {file_name}. Apply modifications based on the overall plan."


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay values take precedence."""
    merged = dict(base)
    for key, val in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def merge_yaml_configs(base_path: str, overlay_path: str, output_path: str) -> None:
    """Deep-merge overlay config into base config, writing result to output_path.
    Overlay keys take precedence; base keys not in overlay are preserved."""
    base_cfg = {}
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                base_cfg = loaded

    overlay_cfg = {}
    if os.path.exists(overlay_path):
        with open(overlay_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                overlay_cfg = loaded

    merged = _deep_merge(base_cfg, overlay_cfg)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)
