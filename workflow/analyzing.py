import json
import os
import re
import hashlib
from tqdm import tqdm
from core.llm_engine import create_client, chat_completion_with_retry
from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.repo_index import (
    collect_context_bundle,
    render_context_bundle_for_prompt,
    summarize_entrypoint_index,
    summarize_repo_index,
)
from core.utils import (print_response, print_log_cost, load_accumulated_cost,
                        save_accumulated_cost,
                        format_paper_content_for_prompt,
                        read_python_files)
import copy
import argparse

logger = get_logger(__name__)


def _prompt_path(prompt_set, name):
    return f"{prompt_set}/{name}" if prompt_set else name


def _format_repo_reference(repo_dir: str, max_total_chars: int = 22000) -> str:
    if not repo_dir or not os.path.isdir(repo_dir):
        return "(none)"

    repo_files = read_python_files(repo_dir)
    if not repo_files:
        return "(none)"

    chunks = []
    total = 0
    for rel_path in sorted(repo_files):
        content = repo_files[rel_path]
        chunk = f"\n## File: {rel_path}\n```python\n{content}\n```\n"
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 400:
                chunks.append(chunk[:remaining] + "\n...(truncated for token budget)...")
            else:
                chunks.append("\n...(truncated for token budget)...")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "".join(chunks).strip() or "(none)"


def _make_safe_artifact_stem(path_text: str) -> str:
    normalized = sanitize_todo_file_name(path_text) or str(path_text or "").strip()
    normalized = normalized.replace("\\", "/")
    readable = normalized.replace("/", "__")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("._")
    readable = re.sub(r"_+", "_", readable)
    if not readable:
        readable = "artifact"
    suffix = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def run_analyzing(paper_name: str, gpt_version: str, output_dir: str,
                  paper_format: str = "JSON",
                  pdf_json_path: str = None,
                  pdf_latex_path: str = None,
                  prompt_set: str = None,
                  baseline_repo_dir: str = None,
                  live_repo_dir: str = None,
                  api_predefine_contract_path: str = None) -> None:

    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = format_paper_content_for_prompt(paper_content, max_chars=22000)
    ctx = load_pipeline_context(output_dir)
    config_yaml = ctx.config_yaml
    context_lst = ctx.context_lst
    overview_prompt = format_paper_content_for_prompt(context_lst[0], max_chars=12000) if len(context_lst) > 0 else ""
    design_prompt = format_paper_content_for_prompt(context_lst[1], max_chars=16000) if len(context_lst) > 1 else ""
    task_prompt = format_paper_content_for_prompt(context_lst[2], max_chars=16000) if len(context_lst) > 2 else ""
    todo_file_lst = ctx.todo_file_lst
    logic_analysis_dict = dict(ctx.logic_analysis_dict)
    repo_index = {
        "repo_manifest": ctx.repo_manifest,
        "symbol_index": ctx.symbol_index,
        "call_graph": ctx.call_graph,
        "entrypoint_index": ctx.entrypoint_index,
    }
    baseline_repo_summary = ""
    entrypoint_summary = ""
    if prompt_set == "feature":
        if ctx.repo_manifest:
            baseline_repo_summary = summarize_repo_index(repo_index, max_chars=18000)
            entrypoint_summary = summarize_entrypoint_index(repo_index, max_chars=5000)
        else:
            baseline_repo_summary = _format_repo_reference(
                live_repo_dir if live_repo_dir and os.path.isdir(live_repo_dir) else baseline_repo_dir
            )
            entrypoint_summary = "(none)"

    done_file_lst = ['config.yaml']

    analysis_msg = [
        {"role": "system", "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_system.txt"),
            paper_format=paper_format)}]

    def _load_feature_file_code(todo_file_name: str):
        candidates = [todo_file_name]
        clean_name = sanitize_todo_file_name(todo_file_name)
        if clean_name and clean_name not in candidates:
            candidates.append(clean_name)
        if prompt_set == "feature" and live_repo_dir:
            for candidate in candidates:
                live_path = os.path.join(live_repo_dir, candidate)
                if os.path.exists(live_path) and os.path.isfile(live_path):
                    with open(live_path, "r", encoding="utf-8") as lf:
                        return lf.read(), "live"
        if prompt_set == "feature" and baseline_repo_dir:
            for candidate in candidates:
                baseline_path = os.path.join(baseline_repo_dir, candidate)
                if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as bf:
                        return bf.read(), "baseline"
        return None, "new"

    def get_write_msg(todo_file_name, todo_file_desc):
        draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
        if len(todo_file_desc.strip()) == 0:
            draft_desc = f"Write the logic analysis in '{todo_file_name}'."

        extra_kwargs = {}
        if prompt_set == "feature":
            extra_kwargs["baseline_repo_summary"] = baseline_repo_summary
            extra_kwargs["repo_manifest_summary"] = baseline_repo_summary
            extra_kwargs["entrypoint_summary"] = entrypoint_summary
            closure = ctx.modification_closure_by_file.get(todo_file_name, {}) or {"path": todo_file_name}
            context_bundle = collect_context_bundle(
                repo_index,
                closure,
                primary_repo_dir=baseline_repo_dir,
                secondary_repo_dir=live_repo_dir,
            ) if ctx.repo_manifest else {}
            bundle_prompt = render_context_bundle_for_prompt(context_bundle) if context_bundle else {}
            if bundle_prompt:
                extra_kwargs.update(bundle_prompt)
                extra_kwargs["baseline_file_code"] = (
                    f"### Source: {bundle_prompt.get('focus_file_source', 'new')}\n"
                    f"{bundle_prompt.get('focus_file_code', '(new file — no baseline/live code)')}"
                )
            else:
                file_code, code_source = _load_feature_file_code(todo_file_name)
                if file_code is None:
                    extra_kwargs["baseline_file_code"] = "(new file — no baseline/live code)"
                    extra_kwargs["focus_file_code"] = "(new file — no baseline/live code)"
                    extra_kwargs["required_context_code"] = "(none)"
                    extra_kwargs["upstream_callers_code"] = "(none)"
                    extra_kwargs["downstream_callees_code"] = "(none)"
                    extra_kwargs["shared_interfaces_code"] = "(none)"
                    extra_kwargs["config_and_registry_code"] = "(none)"
                    extra_kwargs["optional_related_code"] = "(none)"
                    extra_kwargs["entrypoint_chain"] = "(none)"
                    extra_kwargs["synchronized_edit_targets"] = "(none)"
                    extra_kwargs["interface_constraints"] = "(none)"
                    extra_kwargs["target_symbols"] = "(none)"
                else:
                    extra_kwargs["focus_file_code"] = file_code
                    extra_kwargs["baseline_file_code"] = f"### Source: {code_source}\n{file_code}"
                    extra_kwargs["required_context_code"] = "(none)"
                    extra_kwargs["upstream_callers_code"] = "(none)"
                    extra_kwargs["downstream_callees_code"] = "(none)"
                    extra_kwargs["shared_interfaces_code"] = "(none)"
                    extra_kwargs["config_and_registry_code"] = "(none)"
                    extra_kwargs["optional_related_code"] = "(none)"
                    extra_kwargs["entrypoint_chain"] = "(none)"
                    extra_kwargs["synchronized_edit_targets"] = "(none)"
                    extra_kwargs["interface_constraints"] = "(none)"
                    extra_kwargs["target_symbols"] = "(none)"

        write_msg=[{'role': 'user', "content": render_prompt(
            _prompt_path(prompt_set, "analyzing_user.txt"),
            paper_content=paper_content_prompt,
            overview=overview_prompt,
            design=design_prompt,
            task=task_prompt,
            config_yaml=config_yaml,
            draft_desc=draft_desc,
            todo_file_name=todo_file_name,
            **extra_kwargs)}]
        return write_msg

    def api_call(msg):
        return chat_completion_with_retry(client, gpt_version, msg)

    artifact_output_dir = f'{output_dir}/analyzing_artifacts'
    os.makedirs(artifact_output_dir, exist_ok=True)

    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
    for todo_file_name in tqdm(todo_file_lst):
        responses = []
        trajectories = copy.deepcopy(analysis_msg)

        current_stage = f"[ANALYSIS] {todo_file_name}"
        logger.info(current_stage)
        if todo_file_name == "config.yaml":
            continue

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        artifact_stem = _make_safe_artifact_stem(clean_todo_file_name)
        _skip_name = artifact_stem
        _skip_path = os.path.join(artifact_output_dir, f"{_skip_name}_simple_analysis.txt")
        if os.path.exists(_skip_path):
            logger.info(f"  [SKIP] artifact already exists: {_skip_path}")
            done_file_lst.append(clean_todo_file_name)
            continue

        if clean_todo_file_name not in logic_analysis_dict:
            logic_analysis_dict[clean_todo_file_name] = ""
            
        instruction_msg = get_write_msg(clean_todo_file_name, logic_analysis_dict[clean_todo_file_name])
        trajectories.extend(instruction_msg)

        completion = api_call(trajectories)
        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)
        
        message = completion.choices[0].message
        trajectories.append({'role': message.role, 'content': message.content})

        print_response(completion_json)
        temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost

        save_todo_file_name = artifact_stem
        artifact_file_path = os.path.join(artifact_output_dir, f"{save_todo_file_name}_simple_analysis.txt")
        os.makedirs(os.path.dirname(artifact_file_path), exist_ok=True)
        with open(artifact_file_path, 'w') as f:
            f.write(completion_json['choices'][0]['message']['content'])

        done_file_lst.append(clean_todo_file_name)

        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_response.json"), 'w') as f:
            json.dump(responses, f)

        with open(os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_trajectories.json"), 'w') as f:
            json.dump(trajectories, f)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--gpt_version', type=str, default="o3-mini")
    parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument('--pdf_json_path', type=str)
    parser.add_argument('--pdf_latex_path', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    run_analyzing(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
    )
