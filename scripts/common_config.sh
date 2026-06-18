#!/usr/bin/env bash
# Common configuration and functions for math interrupt scripts
# Source this file in your scripts with: source ./common_config.sh

# Common paths and settings


# Common generation parameters
QWEN_MAX_TOKENS=32768
QWEN_TEMPERATURE=0.6
QWEN_TOP_P=0.95
QWEN_TOP_K=20

# GPT-OSS specific parameters
GPTOSS_MAX_TOKENS=131072
GPTOSS_TEMPERATURE=1.0
GPTOSS_TOP_P=1.0
GPTOSS_TOP_K=40

# Mistral specific parameters
MISTRAL_MAX_TOKENS=131072
MISTRAL_TEMPERATURE=0.7
MISTRAL_TOP_P=0.95
MISTRAL_TOP_K=20

# Nemotron specific parameters
NEMOTRON_MAX_TOKENS=131072
NEMOTRON_TEMPERATURE=0.6
NEMOTRON_TOP_P=0.95
NEMOTRON_TOP_K=20

export RUST_LOG="error"
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning"

# Function to setup GPU configuration
setup_gpu_config() {
    local tensor_parallel_size=${1:-1}
    
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -ra GPULIST <<< "$gpu_list"
    TOTAL_GPUS=${#GPULIST[@]}

    # Calculate number of model instances and GPUs per instance
    if [ "$tensor_parallel_size" -gt 1 ]; then
        # Tensor parallelism mode
        GPUS_PER_INSTANCE=$tensor_parallel_size
        NUM_INSTANCES=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
        
        if [ $NUM_INSTANCES -eq 0 ]; then
            echo "Error: Not enough GPUs for tensor parallelism. Need at least $tensor_parallel_size GPUs, but only have $TOTAL_GPUS"
            exit 1
        fi
        
        echo "Tensor parallelism mode: $NUM_INSTANCES instances, $GPUS_PER_INSTANCE GPUs per instance"
    else
        # Data parallelism mode (original behavior)
        NUM_INSTANCES=$TOTAL_GPUS
        GPUS_PER_INSTANCE=1
        echo "Data parallelism mode: $NUM_INSTANCES instances, 1 GPU per instance"
    fi
    
    echo "Launching on ${TOTAL_GPUS} GPUs (${NUM_INSTANCES} instances)"
}

# Function to get GPU list for a specific instance
get_instance_gpus() {
    local idx=$1
    local start_gpu=$((idx * GPUS_PER_INSTANCE))
    local end_gpu=$((start_gpu + GPUS_PER_INSTANCE - 1))
    
    local instance_gpus=""
    for i in $(seq $start_gpu $end_gpu); do
        if [ $i -lt $TOTAL_GPUS ]; then
            if [ -z "$instance_gpus" ]; then
                instance_gpus="${GPULIST[$i]}"
            else
                instance_gpus="$instance_gpus,${GPULIST[$i]}"
            fi
        fi
    done
    echo "$instance_gpus"
}

# Function to run inference instances
run_inference_instances() {
    local script=$1
    local task=$2
    local mode=$3
    local model=$4
    local custom_prompt_file=$5
    local input_file=$6
    local tmp_dir=$7
    local tensor_parallel_size=$8
    local custom_params=${9:-}
    local interrupt_role=${10:-"assistant"} # "assistant" or "user"
    local problem_field_name=${11:-"original_problem"}
    local starter_code_field_name=${12:-"original_starter_code"}
    
    mkdir -p "${tmp_dir}"
    
    # run the script
    for IDX in $(seq 0 $((NUM_INSTANCES-1))); do
        local instance_gpus=$(get_instance_gpus $IDX)
        
        # Build the command as an array
        cmd=(
            python "${script}"
            --model_name "${model}"
            --input_file "${input_file}"
            --output_dir "${tmp_dir}"
            --num_gpus "${NUM_INSTANCES}"
            --local_rank "${IDX}"
            --tensor_parallel_size "${tensor_parallel_size}"
            --mode "${mode}"
            --custom_prompt_file "${custom_prompt_file}"
            --task "${task}"
            --interrupt_role "${interrupt_role}"
            --problem_field_name "${problem_field_name}"
            --starter_code_field_name "${starter_code_field_name}"
        )
        
        # Add custom params if provided
        if [[ -n "${custom_params}" ]]; then
            cmd+=(${custom_params})
        fi

        # Print the full command
        # print if instance_gpus is 0
        if [ "$IDX" -eq 0 ]; then
            echo "Running command: ${cmd[@]}"
        fi

        # Run it
        CUDA_VISIBLE_DEVICES="$instance_gpus" "${cmd[@]}" &
    done
    wait
}

# Function to run interrupt experiments with multiple positions
run_interrupt_experiments() {
    set -euo pipefail

    local script=$1
    local task=$2
    local mode=$3
    local model=$4
    local custom_prompt_file=$5
    local initial_round_file=$6
    local output_dir=$7
    local tmp_dir=$8
    shift 8

    # collect interrupt positions until `--`
    local interrupt_positions=()
    while (($#)); do
        case "$1" in
            --) shift; break;;
            *) interrupt_positions+=("$1"); shift;;
        esac
    done

    # remaining ordered args
    local tensor_parallel_size=${1:-1}; shift || true
    local custom_params=${1:-}; shift || true
    local interrupt_role=${1:-"assistant"}; shift || true
    local problem_field_name=${1:-"original_problem"}; shift || true
    local starter_code_field_name=${1:-"original_starter_code"}; shift || true
    
    for INTERRUPT_POS in "${interrupt_positions[@]}"; do
        # Generate output filename based on model name
        local model_name=$(basename "${model}")
        local final_output="interrupted${INTERRUPT_POS//./}.jsonl"
        
        echo "Running with interrupt_pos=${INTERRUPT_POS}"
        
        # run the script
        for IDX in $(seq 0 $((NUM_INSTANCES-1))); do
            local instance_gpus=$(get_instance_gpus $IDX)

            mkdir -p "${tmp_dir}"

            cmd=(
                python "${script}"
                --model_name "${model}"
                --input_file "${initial_round_file}"
                --output_dir "${tmp_dir}"
                --interrupt_pos "${INTERRUPT_POS}"
                --num_gpus "${NUM_INSTANCES}"
                --local_rank "${IDX}"
                --mode "${mode}"
                --task "${task}"
                --tensor_parallel_size "${tensor_parallel_size}"
                --custom_prompt_file "${custom_prompt_file}"
                --interrupt_role "${interrupt_role}"
                --problem_field_name "${problem_field_name}"
                --starter_code_field_name "${starter_code_field_name}"
                ${custom_params}
            )
            if [ "$IDX" -eq 0 ]; then
                echo "Running command: ${cmd[@]}"
            fi
            CUDA_VISIBLE_DEVICES="$instance_gpus" "${cmd[@]}" &
        done
        wait

        merge_results "${output_dir}" "${final_output}" "${tmp_dir}"
        echo "Finished merging ${final_output}"
    done
}

# Function to merge results
merge_results() {
    local output_dir=$1
    local final_output=$2
    local tmp_dir=$3
    mkdir -p "${output_dir}"
    echo "Merging shards -> ${output_dir}/${final_output}"
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        cat "${tmp_dir}/output_${i}.jsonl"
    done >> "${output_dir}/${final_output}"
    rm -rf "${tmp_dir}"
}

# Function to get common generation parameters
get_qwen_params() {
    echo "--temperature ${QWEN_TEMPERATURE} --top_p ${QWEN_TOP_P} --top_k ${QWEN_TOP_K} --max_tokens ${QWEN_MAX_TOKENS}"
}

# Function to get GPT-OSS specific parameters
get_gptoss_params() {
    echo "--temperature ${GPTOSS_TEMPERATURE} --top_p ${GPTOSS_TOP_P} --top_k ${GPTOSS_TOP_K} --max_tokens ${GPTOSS_MAX_TOKENS}"
}

get_mistral_params() {
    echo "--temperature ${MISTRAL_TEMPERATURE} --top_p ${MISTRAL_TOP_P} --top_k ${MISTRAL_TOP_K} --max_tokens ${MISTRAL_MAX_TOKENS}"
}

get_nemotron_params() {
    echo "--temperature ${NEMOTRON_TEMPERATURE} --top_p ${NEMOTRON_TOP_P} --top_k ${NEMOTRON_TOP_K} --max_tokens ${NEMOTRON_MAX_TOKENS}"
}
