#!/usr/bin/env bash
set -euo pipefail

# Source common configuration (use script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

####################
SCRIPT="src/run.py"
TASK="INSERT"
MODE="INSERT"
CUSTOM_PROMPT_FILE="INSERT"
INTERRUPT_ROLE="INSERT"
INTERRUPT_POS_LIST=(0.1 0.3 0.5 0.7 0.9) # multiple interrupt points
TENSOR_PARALLEL_SIZE=1  # Set to > 1 for tensor parallelism
PROBLEM_FIELD_NAME="INSERT"
STARTER_CODE_FIELD_NAME="INSERT"
####################

####################
# If we use Qwen3-8B, we need to use the following config
MODEL="INSERT"
OUT_DIR="INSERT"
TMP_DIR="INSERT"
INITIAL_ROUND_FILE="INSERT"
PARAMS="$(get_qwen_params)"
####################

####################
# If we use GPT-OSS, we need to use the following config
MODEL="INSERT"
OUT_DIR="INSERT"
TMP_DIR="INSERT"
INITIAL_ROUND_FILE="INSERT"
PARAMS="$(get_gptoss_params)"
####################

# Setup GPU configuration
setup_gpu_config $TENSOR_PARALLEL_SIZE

# Run interrupt experiments using initial round file as input
run_interrupt_experiments \
    "${SCRIPT}" \
    "${TASK}" \
    "${MODE}" \
    "${MODEL}" \
    "${CUSTOM_PROMPT_FILE}" \
    "${INITIAL_ROUND_FILE}" \
    "${OUT_DIR}" \
    "${TMP_DIR}" \
    "${INTERRUPT_POS_LIST[@]}" \
    -- \
    "${TENSOR_PARALLEL_SIZE}" \
    "${PARAMS}" \
    "${INTERRUPT_ROLE}" \
    "${PROBLEM_FIELD_NAME}" \
    "${STARTER_CODE_FIELD_NAME}"
