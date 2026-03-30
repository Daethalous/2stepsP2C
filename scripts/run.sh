#!/bin/bash
# export OPENAI_API_KEY=""

# ============ User Configuration ============

GPT_VERSION="gpt-5-mini"

PAPER_NAME="RECOMBINER"
PDF_JSON_PATH="data/paper2code/paper2code_data/iclr2024/RECOMBINER.json"
OUTPUT_DIR="outputs/RECOMBINER"
GOLD_REPO_DIR="data/paper2code/gold_repos/RECOMBINER"
GENERATED_N=8

# Agent mode: single | baseline | feature | two_step
AGENT_MODE="two_step"

# Output repo directory (the live code repository).
# In two_step mode: auto-derived as "${OUTPUT_DIR}_repo" if left empty.
# In single/baseline/feature modes: must be specified explicitly.
OUTPUT_REPO_DIR=""

# For feature mode only, specify the baseline repo from Step 1:
BASELINE_REPO_DIR=""

# Stages to run (space-separated):
#   preprocess planning extract analyzing coding eval_ref_free eval_ref_based
STAGES="preprocess planning extract analyzing coding eval_ref_free eval_ref_based"

# ============================================

cd "$(dirname "$0")/.."

echo "Paper: ${PAPER_NAME}"
echo "Model: ${GPT_VERSION}"
echo "Agent mode: ${AGENT_MODE}"
echo "Stages: ${STAGES}"
echo ""

CMD="python main.py \
    --paper_name \"${PAPER_NAME}\" \
    --gpt_version \"${GPT_VERSION}\" \
    --pdf_json_path \"${PDF_JSON_PATH}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --gold_repo_dir \"${GOLD_REPO_DIR}\" \
    --generated_n ${GENERATED_N} \
    --agent_mode \"${AGENT_MODE}\" \
    --stages ${STAGES}"

if [ -n "${OUTPUT_REPO_DIR}" ]; then
    CMD="${CMD} --output_repo_dir \"${OUTPUT_REPO_DIR}\""
fi

if [ -n "${BASELINE_REPO_DIR}" ]; then
    CMD="${CMD} --baseline_repo_dir \"${BASELINE_REPO_DIR}\""
fi

eval ${CMD}
