import os
import copy
import random
import json
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from prompt_utils import (
    format_math_initial_input_prompt,
    format_code_generation_initial_input_prompt,
    format_subsequent_input_prompt,
    extract_reasoning_trace,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_tokenizer_and_llm(args):
    if "mistral" in args.model_name.lower():
        tokenizer = MistralTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    hf_overrides = None
    if "Qwen3" in args.model_name:
        if args.max_tokens > 32768:
            original_max_position_embeddings = 32768
            factor = 4.0
            # Use yarn to extend context
            hf_overrides = {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": factor,
                    "original_max_position_embeddings": original_max_position_embeddings,
                },
                "max_model_len": int(original_max_position_embeddings * factor),
            }

    if args.tensor_parallel_size > 1:
        if "mistral" in args.model_name.lower():
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                hf_overrides=hf_overrides,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
            )
        elif "nemotron" in args.model_name.lower():
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                hf_overrides=hf_overrides,
                mamba_ssm_cache_dtype="float32",
                trust_remote_code=True,
            )
        else:
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                hf_overrides=hf_overrides,
            )
    else:
        if "mistral" in args.model_name.lower():
            llm = LLM(
                model=args.model_name,
                hf_overrides=hf_overrides,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
            )
        elif "nemotron" in args.model_name.lower():
            llm = LLM(
                model=args.model_name,
                hf_overrides=hf_overrides,
                mamba_ssm_cache_dtype="float32",
                trust_remote_code=True,
            )
        else:
            llm = LLM(model=args.model_name, hf_overrides=hf_overrides)
    return tokenizer, llm


########################################
# Inference utils
########################################


def inference_loop(args, data, llm, tokenizer, initial_round=False):
    metadata = [
        {"enforce_thinking_budget": args.enforce_thinking_budget}
        for _ in range(len(data))
    ]

    outputs = llm.generate(
        [x["formatted_input_prompt"][-1] for x in data],
        sampling_params=SamplingParams(
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ),
    )
    outputs_tokenized = [output.outputs[0].token_ids for output in outputs]
    outputs_text = [
        tokenizer.decode(output_tokenized) for output_tokenized in outputs_tokenized
    ]

    for idx in range(len(data)):
        reasoning_trace = extract_reasoning_trace(
            outputs_text[idx], model_name=args.model_name
        )
        total_reasoning_length = len(tokenizer.encode(reasoning_trace))
        data[idx]["output"].append(outputs_text[idx])
        if initial_round:
            metadata[idx]["total_reasoning_length"] = total_reasoning_length
        data[idx]["metadata"].append(metadata[idx])

    return data


def run_initial_round(args, data, custom_prompts, tokenizer, llm):
    for i, ex in enumerate(tqdm(data, total=len(data), desc=f"Rank {args.local_rank}")):
        data[i]["metadata"] = []
        data[i]["output"] = []
        # input formatting
        if args.task == "math":
            data[i]["formatted_input_prompt"] = [
                format_math_initial_input_prompt(
                    args.model_name,
                    ex,
                    custom_prompts,
                    tokenizer,
                    args.mode,
                    args.problem_field_name,
                )
            ]
        elif args.task == "code":
            data[i]["formatted_input_prompt"] = [
                format_code_generation_initial_input_prompt(
                    args.model_name,
                    ex,
                    custom_prompts,
                    tokenizer,
                    args.mode,
                    args.problem_field_name,
                )
            ]

    # generate outputs
    data = inference_loop(args, data, llm, tokenizer, initial_round=True)
    return data


def run_subsequent_intervene(args, data, custom_prompts, tokenizer, llm):
    for idx, ex in enumerate(
        tqdm(data, total=len(data), desc=f"Round {round} | Rank {args.local_rank}")
    ):
        # input formatting (starting from the formatted input prompt of the previous round)
        data[idx]["formatted_input_prompt"].append(
            format_subsequent_input_prompt(args, ex, custom_prompts, tokenizer)
        )

    data = inference_loop(args, data, llm, tokenizer, initial_round=False)
    return data


########################################
# Data Utils
########################################


def copy_data(data, data_len):
    copied_data = copy.deepcopy(data)
    for ex in copied_data:
        ex["id"] = ex["id"] + data_len
    return copied_data


def load_data(args):
    # Load data
    if os.path.exists(args.input_file):
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
    else:
        hf_data = load_dataset(args.input_file, split="train")
        data = [dict(example) for example in hf_data]

    if args.task == "math":
        # duplicate the data
        if "initial" in args.mode:
            extra_copies = [
                ("gsm8k", 0),
                ("math500", 0),
                ("aime2024", 15),
                ("aime2025", 15),
            ]
            for source, num_copies in extra_copies:
                subset_data = [ex for ex in data if ex["source"] == source]
                for _ in range(num_copies):
                    copied_subset_data = copy_data(subset_data, len(data))
                    data.extend(copied_subset_data)
        data = data[args.local_rank :: args.num_gpus]

    elif args.task == "code":
        data = data[args.local_rank :: args.num_gpus]

    # Load custom prompts if provided
    if args.custom_prompt_file is not None:
        with open(args.custom_prompt_file, "r") as f:
            custom_prompts = json.load(f)
    else:
        custom_prompts = {}
    return data, custom_prompts


def save_data(args, data):
    rank = args.local_rank
    output_file = os.path.join(args.output_dir, f"output_{rank}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
