import argparse
import copy
import json
import os

from tqdm import tqdm

from core.data_loader import load_paper_content, load_pipeline_context, sanitize_todo_file_name
from core.llm_engine import (
    chat_completion_with_retry,
    create_client,
    prepare_messages_for_api,
    sanitize_prompt_text,
)
from core.logger import get_logger
from core.prompts.templates import render_prompt
from core.repo_index import (
    collect_context_bundle,
    render_context_bundle_for_prompt,
    summarize_entrypoint_index,
    summarize_repo_index,
)
from core.utils import (
    format_paper_content_for_prompt,
    load_accumulated_cost,
    load_baseline_interface_stub_text,
    print_log_cost,
    print_response,
    read_python_files,
    save_accumulated_cost,
)
from workflow.feature_agent.rpg_adapter import (
    get_feature_analysis_context,
    get_feature_file_order,
    load_baseline_analysis_dict,
    load_or_build_feature_rpg_bundle,
    make_safe_artifact_stem,
)

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


def run_rpg_analyzing(
    paper_name: str,
    gpt_version: str,
    output_dir: str,
    paper_format: str = "JSON",
    pdf_json_path: str = None,
    pdf_latex_path: str = None,
    prompt_set: str = None,
    baseline_repo_dir: str = None,
    live_repo_dir: str = None,
    baseline_interface_stub_path: str = None,
) -> None:
    client = create_client()
    paper_content = load_paper_content(paper_format, pdf_json_path, pdf_latex_path)
    paper_content_prompt = sanitize_prompt_text(
        format_paper_content_for_prompt(paper_content, max_chars=22000),
        max_chars=22000,
    )
    ctx = load_pipeline_context(output_dir)
    config_yaml = sanitize_prompt_text(ctx.config_yaml, max_chars=16000)
    context_lst = ctx.context_lst
    overview_prompt = sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[0], max_chars=12000),
        max_chars=12000,
    ) if len(context_lst) > 0 else ""
    design_prompt = sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[1], max_chars=16000),
        max_chars=16000,
    ) if len(context_lst) > 1 else ""
    task_prompt = sanitize_prompt_text(
        format_paper_content_for_prompt(context_lst[2], max_chars=16000),
        max_chars=16000,
    ) if len(context_lst) > 2 else ""
    repo_index = {
        "repo_manifest": ctx.repo_manifest,
        "symbol_index": ctx.symbol_index,
        "call_graph": ctx.call_graph,
        "entrypoint_index": ctx.entrypoint_index,
    }

    rpg, file_metadata = load_or_build_feature_rpg_bundle(
        output_dir=output_dir,
        baseline_interface_stub_path=baseline_interface_stub_path,
    )
    todo_file_lst = get_feature_file_order(rpg, ctx.todo_file_lst, file_metadata)
    completed_analysis_dict = {}
    baseline_analysis_dict = load_baseline_analysis_dict(
        baseline_interface_stub_path=baseline_interface_stub_path,
        todo_file_lst=todo_file_lst,
    )

    baseline_repo_summary = ""
    entrypoint_summary = ""
    baseline_interface_stub = "(none)"
    if prompt_set == "feature":
        if ctx.repo_manifest:
            baseline_repo_summary = sanitize_prompt_text(
                summarize_repo_index(repo_index, max_chars=18000),
                max_chars=18000,
            )
            entrypoint_summary = sanitize_prompt_text(
                summarize_entrypoint_index(repo_index, max_chars=5000),
                max_chars=5000,
            )
        else:
            baseline_repo_summary = sanitize_prompt_text(
                _format_repo_reference(
                    live_repo_dir if live_repo_dir and os.path.isdir(live_repo_dir) else baseline_repo_dir
                ),
                max_chars=22000,
            )
            entrypoint_summary = "(none)"
        baseline_interface_stub = sanitize_prompt_text(load_baseline_interface_stub_text(
            explicit_path=baseline_interface_stub_path,
            max_chars=10000,
        ), max_chars=10000)

    analysis_msg = [
        {
            "role": "system",
            "content": sanitize_prompt_text(
                render_prompt(_prompt_path(prompt_set, "analyzing_system.txt"), paper_format=paper_format)
            ),
        }
    ]

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

    def get_write_msg(todo_file_name: str, todo_file_desc: str):
        todo_file_name = sanitize_prompt_text(todo_file_name)
        todo_file_desc = sanitize_prompt_text(todo_file_desc, max_chars=12000)
        draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
        if len(todo_file_desc.strip()) == 0:
            draft_desc = f"Write the logic analysis in '{todo_file_name}'."

        dep_context = sanitize_prompt_text(get_feature_analysis_context(
            rpg=rpg,
            target_file=todo_file_name,
            completed_analysis_dict=completed_analysis_dict,
            baseline_analysis_dict=baseline_analysis_dict,
            max_chars=4000,
        ), max_chars=4000)
        if dep_context:
            deps = rpg.get_dependencies(todo_file_name)
            draft_desc += (
                f"\n\n## Upstream Dependencies\n"
                f"This file depends on: {deps}\n"
                "Below are the logic analyses of dependency files. "
                "Your analysis must stay consistent with their interfaces and responsibilities.\n\n"
                f"{dep_context}"
            )

        extra_kwargs = {}
        if prompt_set == "feature":
            extra_kwargs["baseline_repo_summary"] = baseline_repo_summary
            extra_kwargs["repo_manifest_summary"] = baseline_repo_summary
            extra_kwargs["entrypoint_summary"] = entrypoint_summary
            extra_kwargs["baseline_interface_stub"] = baseline_interface_stub
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
                else:
                    extra_kwargs["focus_file_code"] = file_code
                    extra_kwargs["baseline_file_code"] = f"### Source: {code_source}\n{file_code}"
                for key in (
                    "required_context_code",
                    "upstream_callers_code",
                    "downstream_callees_code",
                    "shared_interfaces_code",
                    "config_and_registry_code",
                    "optional_related_code",
                    "entrypoint_chain",
                    "synchronized_edit_targets",
                    "interface_constraints",
                    "target_symbols",
                    "repo_primary_entrypoints",
                ):
                    extra_kwargs.setdefault(key, "(none)")

        draft_desc = sanitize_prompt_text(draft_desc, max_chars=20000)
        extra_kwargs = {key: sanitize_prompt_text(value, max_chars=22000) for key, value in extra_kwargs.items()}

        return [
            {
                "role": "user",
                "content": sanitize_prompt_text(render_prompt(
                    _prompt_path(prompt_set, "analyzing_user.txt"),
                    paper_content=paper_content_prompt,
                    overview=overview_prompt,
                    design=design_prompt,
                    task=task_prompt,
                    config_yaml=config_yaml,
                    draft_desc=draft_desc,
                    todo_file_name=todo_file_name,
                    **extra_kwargs,
                )),
            }
        ]

    def api_call(msg):
        safe_messages, _, payload_len = prepare_messages_for_api(msg, model=gpt_version)
        logger.info(
            "[FEATURE_RPG_ANALYSIS] Prepared payload msg_count=%s payload_len=%s",
            len(safe_messages),
            payload_len,
        )
        return chat_completion_with_retry(client, gpt_version, safe_messages)

    artifact_output_dir = os.path.join(output_dir, "analyzing_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)
    total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")

    for todo_file_name in tqdm(todo_file_lst):
        responses = []
        trajectories = copy.deepcopy(analysis_msg)
        current_stage = f"[FEATURE_RPG_ANALYSIS] {todo_file_name}"
        logger.info(current_stage)

        if todo_file_name == "config.yaml":
            continue

        clean_todo_file_name = sanitize_todo_file_name(todo_file_name)
        artifact_stem = make_safe_artifact_stem(clean_todo_file_name)
        artifact_file_path = os.path.join(artifact_output_dir, f"{artifact_stem}_simple_analysis.txt")
        legacy_artifact_file_path = os.path.join(
            artifact_output_dir, f"{clean_todo_file_name.replace('/', '_')}_simple_analysis.txt"
        )
        if os.path.exists(artifact_file_path) or os.path.exists(legacy_artifact_file_path):
            existing_path = artifact_file_path if os.path.exists(artifact_file_path) else legacy_artifact_file_path
            logger.info(f"  [SKIP] artifact already exists: {existing_path}")
            with open(existing_path, "r", encoding="utf-8") as f:
                completed_analysis_dict[clean_todo_file_name] = sanitize_prompt_text(f.read())
            continue

        instruction_msg = get_write_msg(clean_todo_file_name, str(ctx.logic_analysis_dict.get(clean_todo_file_name, "")))
        trajectories.extend(instruction_msg)
        completion = api_call(trajectories)
        completion_json = json.loads(completion.model_dump_json())
        responses.append(completion_json)

        message = completion.choices[0].message
        model_text = sanitize_prompt_text(message.content)
        trajectories.append({"role": message.role, "content": model_text})
        completed_analysis_dict[clean_todo_file_name] = model_text

        print_response(completion_json)
        total_accumulated_cost = print_log_cost(
            completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost
        )

        with open(artifact_file_path, "w", encoding="utf-8") as f:
            f.write(model_text)
        with open(
            os.path.join(output_dir, f"{artifact_stem}_simple_analysis_response.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(responses, f)
        with open(
            os.path.join(output_dir, f"{artifact_stem}_simple_analysis_trajectories.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(trajectories, f)

    save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_name", type=str)
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    parser.add_argument("--paper_format", type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--pdf_json_path", type=str)
    parser.add_argument("--pdf_latex_path", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--baseline_repo_dir", type=str, default="")
    parser.add_argument("--live_repo_dir", type=str, default="")
    parser.add_argument("--baseline_interface_stub", type=str, default="")
    args = parser.parse_args()

    run_rpg_analyzing(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        paper_format=args.paper_format,
        pdf_json_path=args.pdf_json_path,
        pdf_latex_path=args.pdf_latex_path,
        prompt_set="feature",
        baseline_repo_dir=args.baseline_repo_dir,
        live_repo_dir=args.live_repo_dir,
        baseline_interface_stub_path=args.baseline_interface_stub or None,
    )
