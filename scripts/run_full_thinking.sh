#!/usr/bin/env bash
set -euo pipefail

# Source common configuration (use script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

####################
SCRIPT="src/run.py"
TASK="INSERT"
MODE="initial"
INPUT_FILE="INSERT"
CUSTOM_PROMPT_FILE="INSERT"
INTERRUPT_ROLE="INSERT"
PROBLEM_FIELD_NAME="original_problem"
STARTER_CODE_FIELD_NAME="original_starter_code"
TENSOR_PARALLEL_SIZE=1  # Set to > 1 for tensor parallelism
####################

####################
# If we use Qwen3-8B, we need to use the following config
MODEL="INSERT"
FINAL_OUT="INSERT"
OUT_DIR="INSERT"
TMP_DIR="INSERT"
PARAMS="$(get_qwen_params)"
####################

####################
# If we use GPT-OSS, we need to use the following config
MODEL="INSERT"
FINAL_OUT="INSERT"
OUT_DIR="INSERT"
TMP_DIR="INSERT"
PARAMS="$(get_gptoss_params)"
####################

# Setup GPU configuration
setup_gpu_config $TENSOR_PARALLEL_SIZE

# Run inference with common parameters
run_inference_instances \
    "${SCRIPT}" \
    "${TASK}" \
    "${MODE}" \
    "${MODEL}" \
    "${CUSTOM_PROMPT_FILE}" \
    "${INPUT_FILE}" \
    "${TMP_DIR}" \
    "${TENSOR_PARALLEL_SIZE}" \
    "${PARAMS}" \
    "${INTERRUPT_ROLE}" \
    "${PROBLEM_FIELD_NAME}" \
    "${STARTER_CODE_FIELD_NAME}"

# Merge results
merge_results "${OUT_DIR}" "${FINAL_OUT}" "${TMP_DIR}"
