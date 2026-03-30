import json
import os
import argparse
from core.exceptions import PipelineError
from core.llm_engine import create_client, chat_completion_raw_with_retry
from core.logger import get_logger
from core.utils import read_python_files, extract_planning, content_to_json, \
        num_tokens_from_messages, read_all_files, extract_json_from_string, get_now_str, print_log_cost

logger = get_logger(__name__)


def run_eval(paper_name: str, gpt_version: str, output_dir: str,
             pdf_json_path: str, target_repo_dir: str,
             eval_result_dir: str, eval_type: str = "ref_free",
             prompts_dir: str = None,
             gold_repo_dir: str = "",
             generated_n: int = 8,
             is_papercoder: bool = True,
             selected_file_path: str = "") -> None:

    if prompts_dir is None:
        prompts_dir = os.path.dirname(os.path.abspath(__file__))

    client = create_client()

    def api_call(request_json):
        return chat_completion_raw_with_retry(client, request_json)

    with open(f'{pdf_json_path}') as f:
        paper_json = json.load(f)

    codes = ""
    if is_papercoder:
        target_files_dict = read_python_files(target_repo_dir)

        with open(f'{output_dir}/planning_config.yaml') as f:
            config_yaml = f.read()

        context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

        if os.path.exists(f'{output_dir}/task_list.json'):
            with open(f'{output_dir}/task_list.json') as f:
                task_list = json.load(f)
        else:
            task_list = content_to_json(context_lst[2])

        for file_name, file_content in target_files_dict.items():
            codes += f"```python\n## File name: {file_name}\n{file_content}\n```\n\n"

        codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"
    else:
        target_files_dict = read_all_files(target_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)
        for file_name, code in target_files_dict.items():
            codes += f"```## File name: {file_name}\n{code}\n```\n\n"

    prompt = open(f"{prompts_dir}/{eval_type}.txt").read()

    cur_prompt = prompt.replace('{{Paper}}', f"{paper_json}").replace('{{Code}}', codes)

    if "ref_based" == eval_type and len(gold_repo_dir) > 0:
        all_files_dict = read_all_files(gold_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)

        goldcodes = ""
        gold_cnt = 0
        if len(selected_file_path) > 0:
            selected_file_lst = []
            with open(selected_file_path) as f:
                selected_file_lst = f.readlines()

            for s_idx in range(len(selected_file_lst)):
                selected_file_lst[s_idx] = selected_file_lst[s_idx].strip()

            for all_file, all_file_code in all_files_dict.items():
                if all_file not in selected_file_lst:
                    continue
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1
        else:
            for all_file, all_file_code in all_files_dict.items():
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1

        cur_prompt = cur_prompt.replace('{{GoldCode}}', f"{goldcodes}")

    msg = [{"role": "system", "content": cur_prompt}]

    try:
        num_tokens = num_tokens_from_messages(msg)
    except Exception as e:
        logger.warning(f"[WARNING] An exception was raised while counting tokens for the target repository of {paper_name}.")
        logger.warning(e)
        logger.warning("-"*40)
        num_tokens = 0

    if num_tokens > 1000000:
        raise PipelineError(f"{paper_name} exceeds 1000k token limit ({num_tokens} tokens)")

    if "o3-mini" in gpt_version:
        if generated_n > 8:
            logger.warning(f"[WARNING] o3-mini does not support n > 8. Setting generated_n to 8.")
            generated_n = 8

        request_json = {
                "model": gpt_version,
                "messages": msg,
                "reasoning_effort": "high",
                "n": generated_n
        }
    else:
        request_json = {
                "model": gpt_version,
                "messages": msg,
                "temperature": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "n": generated_n
        }

    completion = api_call(request_json)
    completion_json = json.loads(completion.model_dump_json())

    score_key = "score"
    rationale_key = "critique_list"

    all_scores = []
    rationales = []
    for n in range(generated_n):
        choice = completion_json['choices'][n]
        output = choice['message']['content'].strip()

        try:
            output_json2 = json.loads(output)
            score = int(output_json2[score_key])

            if isinstance(output_json2[rationale_key], str):
                rationale = output_json2[rationale_key]
            else:
                rationale = json.dumps(output_json2[rationale_key])
        except Exception as e:
            try:
                output_json2 = json.loads(extract_json_from_string(output))
                score = int(output_json2[score_key])

                if isinstance(output_json2[rationale_key], str):
                    rationale = output_json2[rationale_key]
                else:
                    rationale = json.dumps(output_json2[rationale_key])
            except Exception as e2:
                logger.warning(f"[WARNING] Invalid repsponse: parsing error")
                logger.warning(e2)
                logger.warning("-"*40)
                continue

        if score < 1 or score > 5:
            logger.warning(f"[WARNING] Invalid repsponse: score {score}, Score must be in the range of 1-5.")
            continue

        all_scores.append(int(score))
        rationales.append(rationale)

    if len(all_scores) == 0:
        logger.error("[ERROR] No valid scores could be parsed from the model responses.")
        avg_score = 0
    else:
        avg_score = sum(all_scores) / len(all_scores)

    output_json = {
        "paper_name": paper_name,
        "target_repo_dir": target_repo_dir,
        "eval_type": eval_type,
        "gold_repo_dir": gold_repo_dir,
        "generated_n": generated_n,
        "request_json": request_json,
        "completion_json": completion_json,
        "eval_result": {
            "score": avg_score,
            "valid_n": len(all_scores),
            "scroe_lst": all_scores,
            "rationale_lst": rationales,
        },
    }

    now_str = get_now_str()
    os.makedirs(eval_result_dir, exist_ok=True)
    with open(f"{eval_result_dir}/{paper_name}_eval_{eval_type}_{gpt_version}_{now_str}.json", 'w', encoding='utf-8') as f:
        json.dump(output_json, f)

    logger.info("")
    logger.info("=" * 40)
    logger.info("Evaluation Summary")
    logger.info(f"  Paper name: {paper_name}")
    logger.info(f"  Evaluation type: {eval_type}")
    logger.info(f"  Target repo directory: {target_repo_dir}")
    logger.info(f"  Evaluation result:")
    logger.info(f"    Score: {avg_score:.4f}")
    logger.info(f"    Valid: {output_json['eval_result']['valid_n']}/{generated_n}")
    logger.info("=" * 40)

    print_log_cost(completion_json, gpt_version, f"[Evaluation] {paper_name} - {eval_type}", output_dir, 0)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--paper_name', type=str)
    argparser.add_argument('--pdf_json_path', type=str)
    argparser.add_argument('--prompts_dir', type=str, default=None,
                           help="Directory containing ref_free.txt and ref_based.txt prompt files")

    argparser.add_argument('--output_dir', type=str)

    argparser.add_argument('--target_repo_dir', type=str)
    argparser.add_argument('--gold_repo_dir', type=str, default="")
    argparser.add_argument('--eval_result_dir', type=str)

    argparser.add_argument('--eval_type', type=str, default="ref_free", choices=["ref_free", "ref_based"])

    argparser.add_argument('--generated_n', type=int, default=8)
    argparser.add_argument('--gpt_version', type=str, default="o3-mini")

    argparser.add_argument('--selected_file_path', type=str, default="")
    argparser.add_argument('--papercoder', action="store_true")

    args = argparser.parse_args()

    run_eval(
        paper_name=args.paper_name,
        gpt_version=args.gpt_version,
        output_dir=args.output_dir,
        pdf_json_path=args.pdf_json_path,
        target_repo_dir=args.target_repo_dir,
        eval_result_dir=args.eval_result_dir,
        eval_type=args.eval_type,
        prompts_dir=args.prompts_dir,
        gold_repo_dir=args.gold_repo_dir,
        generated_n=args.generated_n,
        is_papercoder=True if args.papercoder else False,
        selected_file_path=args.selected_file_path,
    )
