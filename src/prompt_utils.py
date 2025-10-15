# logic and formatting adapted from LiceCodeBench repo
# https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py

from huggingface_hub import hf_hub_download
from vllm.transformers_utils.tokenizers.mistral import (
    make_mistral_chat_completion_request,
)

SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

GPTOSS_END_THINKING_TOKEN = "<|end|>"
GPTOSS_END_ANSWER_TOKEN = "<|return|>"
QWEN_END_THINKING_TOKEN = "</think>"
QWEN_END_ANSWER_TOKEN = "<|im_end|>"


def get_code_generation_messages_qwen(
    entry, custom_prompts, problem_field_name, starter_code_field_name
):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{entry[problem_field_name]}\n\n"
    if entry[starter_code_field_name]:
        prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{entry[starter_code_field_name]}\n```\n\n"
    else:
        prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    system_prompt = (
        custom_prompts.get("system_prompt", "") + "\n" + SYSTEM_MESSAGE_GENERIC
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


def get_code_generation_messages_openaioss(
    entry, custom_prompts, problem_field_name, starter_code_field_name
):
    prompt = f"### Question:\n{entry[problem_field_name]}\n\n"
    if entry[starter_code_field_name]:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{entry[starter_code_field_name]}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    system_prompt = (
        custom_prompts.get("system_prompt", "") + "\n" + SYSTEM_MESSAGE_GENERIC
    )
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


def load_mistral_system_prompt_code(custom_prompts):
    final_piece = "Here, provide a self-contained response."
    if custom_prompts.get("system_prompt", "") != "":
        final_block = custom_prompts.get("system_prompt", "") + "\n" + final_piece
    else:
        final_block = final_piece
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_MESSAGE_GENERIC
                + " First draft your thinking process (inner monologue) until you arrive at a response. \n\nYour thinking process must follow the template below:",
            },
            {
                "type": "thinking",
                "thinking": "Your thoughts or/and scratch work for designing and implementing the solution. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.",
                "closed": True,
            },
            {
                "type": "text",
                "text": final_block,
            },
        ],
    }


def get_code_generation_messages_mistral(
    entry, custom_prompts, model_name, problem_field_name, starter_code_field_name
):
    prompt = f"### Question:\n{entry[problem_field_name]}\n\n"
    if entry[starter_code_field_name]:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{entry[starter_code_field_name]}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    # Load the system prompt from Mistral
    mistral_system_prompt = load_mistral_system_prompt_code(custom_prompts)
    messages = [
        mistral_system_prompt,
        {"role": "user", "content": prompt},
    ]
    # system_prompt = custom_prompts.get("system_prompt", "") + "\n" + SYSTEM_MESSAGE_GENERIC

    return messages


def prompt_formatting_model(model_name, tokenizer, messages):
    if "qwen3" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    elif "deepseek-r1" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "gpt-oss" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high",
        )
    elif "mistral" in model_name.lower():
        request = make_mistral_chat_completion_request(messages, tools=None)
        encoded = tokenizer.mistral.encode_chat_completion(request)
        # Patch for Magistral as they won't always enable thinking...
        formatted_prompt = encoded.text + " [THINK]"
    else:
        raise ValueError(f"Model {model_name} not supported")
    return formatted_prompt


def extract_reasoning_trace(output_text: str, model_name: str):
    if "gpt-oss" in model_name.lower():
        # Format: <|channel|>analysis<|message|>...<|end|>...<|channel|>final<|message|>...<|return|>
        if "<|channel|>analysis<|message|>" in output_text:
            # between the <|channel|>analysis<|message|> and <|end|>
            reasoning_trace = output_text.split("<|end|>")[0]
        else:
            reasoning_trace = output_text
    elif "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        if "</think>" in output_text:
            reasoning_trace = output_text.split("</think>")[0]
        else:
            reasoning_trace = output_text
    elif "mistral" in model_name.lower():
        if "[/THINK]" in output_text:
            reasoning_trace = output_text.split("[/THINK]")[0]
        else:
            reasoning_trace = output_text
    return reasoning_trace


def thinking_token_check(model_name, reasoning_trace):
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        if "<think>\n" not in reasoning_trace:
            reasoning_trace = "<think>\n" + reasoning_trace
    elif "gpt-oss" in model_name.lower():
        if "<|channel|>analysis<|message|>" not in reasoning_trace:
            reasoning_trace = "<|channel|>analysis<|message|>\n" + reasoning_trace
    elif "mistral" in model_name.lower():
        if "[/THINK]" not in reasoning_trace:
            reasoning_trace = "[/THINK]\n" + reasoning_trace
    return reasoning_trace


def get_generation_status(model_name, outputs):
    if "gpt-oss" in model_name.lower():
        end_thinking_token = GPTOSS_END_THINKING_TOKEN
        end_answer_token = GPTOSS_END_ANSWER_TOKEN
    elif "qwen3" in model_name.lower():
        end_thinking_token = QWEN_END_THINKING_TOKEN
        end_answer_token = QWEN_END_ANSWER_TOKEN
    else:
        raise ValueError(f"Model {model_name} not supported")
    still_thinking_indices, still_answering_indices, done_indices = [], [], []
    for idx, output in enumerate(outputs):
        if end_thinking_token not in output:
            still_thinking_indices.append(idx)
        elif end_answer_token not in output:
            still_answering_indices.append(idx)
        else:
            done_indices.append(idx)
    return still_thinking_indices, still_answering_indices, done_indices


def truncate_reasoning_trace(model_name, reasoning_trace, tokenizer, interrupt_pos):
    reasoning_trace_ids = tokenizer.encode(reasoning_trace)
    reasoning_trace_ids = reasoning_trace_ids[:interrupt_pos]
    reasoning_trace = tokenizer.decode(reasoning_trace_ids)
    # We need to enssure that the open bracket is not removed
    reasoning_trace = thinking_token_check(model_name, reasoning_trace)
    return reasoning_trace


def close_reasoning_trace(model_name, reasoning_trace, force_answer=False):
    assert any(
        m in model_name.lower() for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        if force_answer:
            reasoning_trace += "\n</think>\n\n"
        else:
            reasoning_trace += "\n</think>\n\n<|im_end|>"
    elif "gpt-oss" in model_name.lower():
        if force_answer:
            # End the current thinking channel and then start the final (answer) channel
            reasoning_trace += "<|end|><|start|>assistant<|channel|>final<|message|>"
        else:
            # End the current thinking channel
            reasoning_trace += "<|end|>"
    elif "mistral" in model_name.lower():
        if force_answer:
            reasoning_trace += "\n\n[/THINK]"
        else:
            reasoning_trace += "\n\n[/THINK]</s>"
    return reasoning_trace


def open_close_reasoning_trace(model_name):
    assert any(
        m in model_name.lower() for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        return "<think>\n\n</think>\n\n"
    elif "gpt-oss" in model_name.lower():
        return "<|channel|>analysis<|message|>\n<|end|><|start|>assistant<|channel|>final<|message|>"
    elif "mistral" in model_name.lower():
        return "[THINK]\n\n[/THINK]"


def format_user_turn(model_name, update_instructions):
    assert any(
        m in model_name.lower() for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        user_turn = f"\n<|im_start|>user\n{update_instructions}\n<|im_end|>\n<|im_start|>assistant\n"
    elif "gpt-oss" in model_name.lower():
        user_turn = (
            f"<|start|>user<|message|>{update_instructions}<|end|><|start|>assistant"
        )
    elif "mistral" in model_name.lower():
        user_turn = f"[INST]{update_instructions}[/INST]"
    return user_turn


# directly copied from https://github.com/openai/gpt-oss
AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""


def get_math_messages_qwen(ex, custom_prompts, mode, problem_field_name):
    prompt = "Q: " + ex[problem_field_name] + "\n" + "A:"
    prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        + prompt
    )
    messages = [{"role": "user", "content": prompt}]
    if custom_prompts is not None and custom_prompts.get("system_prompt", "") != "":
        messages = [
            {"role": "system", "content": custom_prompts["system_prompt"]}
        ] + messages
    return messages


def load_mistral_system_prompt(repo_id: str, filename: str):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


def get_math_messages_mistral(ex, custom_prompts, mode, problem_field_name):
    mistral_system_prompt = load_mistral_system_prompt(
        "mistralai/Magistral-Small-2509", "SYSTEM_PROMPT.txt"
    )
    mistral_system_prompt["content"][0]["text"] = (
        custom_prompts.get("system_prompt", "")
        + "\n"
        + mistral_system_prompt["content"][0]["text"]
    )
    messages = [
        mistral_system_prompt,
        {
            "role": "user",
            "content": AIME_TEMPLATE.format(question=ex[problem_field_name]),
        },
    ]
    return messages


def get_math_messages_openaioss(ex, custom_prompts, mode, problem_field_name):
    question = ex[problem_field_name]
    # in a conservative manner (i don't wanna add potential bugs)
    prompt = AIME_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    # developer role is used for GPT-OSS
    # Reference: https://cookbook.openai.com/articles/openai-harmony
    if custom_prompts is not None and custom_prompts.get("system_prompt", "") != "":
        messages = [
            {"role": "developer", "content": custom_prompts["system_prompt"]}
        ] + messages
    return messages


def format_math_initial_input_prompt(
    model_name, ex, custom_prompts, tokenizer, mode, problem_field_name="problem"
):
    # Step 1: Get the messages
    if "qwen3" in model_name.lower():
        messages = get_math_messages_qwen(ex, custom_prompts, mode, problem_field_name)
    elif "gpt-oss" in model_name.lower():
        messages = get_math_messages_openaioss(
            ex, custom_prompts, mode, problem_field_name
        )
    elif "mistral" in model_name.lower():
        messages = get_math_messages_mistral(
            ex, custom_prompts, mode, problem_field_name
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    # Step 2: Format the prompt (with specific formatting for different models)
    formatted_prompt = prompt_formatting_model(model_name, tokenizer, messages)
    return formatted_prompt


def format_code_generation_initial_input_prompt(
    model_name,
    ex,
    custom_prompts,
    tokenizer,
    mode,
    problem_field_name="problem",
    starter_code_field_name="starter_code",
):
    if "qwen3" in model_name.lower():
        messages = get_code_generation_messages_qwen(
            ex, custom_prompts, problem_field_name, starter_code_field_name
        )
    elif "gpt-oss" in model_name.lower():
        messages = get_code_generation_messages_openaioss(
            ex, custom_prompts, problem_field_name, starter_code_field_name
        )
    elif "mistral" in model_name.lower():
        messages = get_code_generation_messages_mistral(
            ex, custom_prompts, model_name, problem_field_name, starter_code_field_name
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    formatted_prompt = prompt_formatting_model(model_name, tokenizer, messages)
    return formatted_prompt


def format_subsequent_input_prompt(args, ex, custom_prompts, tokenizer):
    # Step 1: Format update instructions
    update_instructions = ""
    if args.mode == "subsequent_intervene" or args.mode == "subsequent_intervene_hard":
        update_instructions = ex["update"]
        update_prefix = custom_prompts.get("update_prefix", "")
        update_suffix = custom_prompts.get("update_suffix", "")
        update_instructions = update_prefix + update_instructions + update_suffix

    # Step 2: Truncate reasoning trace
    formatted_prompt = ex["formatted_input_prompt"][-1]
    total_reasoning_length = ex["metadata"][0]["total_reasoning_length"]
    reasoning_trace = extract_reasoning_trace(ex["output"][-1], args.model_name)
    if int(args.interrupt_pos) == 1:
        # We don't truncate the reasoning trace for full reasoning trace cases
        truncated_reasoning_trace = reasoning_trace
        if total_reasoning_length > 130000:
            truncated_reasoning_trace = truncate_reasoning_trace(
                args.model_name, reasoning_trace, tokenizer, 130000
            )
    else:
        interrupt_pos = int(args.interrupt_pos * total_reasoning_length)
        truncated_reasoning_trace = truncate_reasoning_trace(
            args.model_name, reasoning_trace, tokenizer, interrupt_pos
        )

    # Step 3: Append update instructions
    if args.interrupt_role == "assistant":
        prompts_prefix = custom_prompts.get("reasoning_prefix", "")
        prompts_suffix = custom_prompts.get("reasoning_suffix", "")
        update_instructions = prompts_prefix + update_instructions + prompts_suffix
        updated_reasoning_trace = truncated_reasoning_trace + update_instructions
        # Add the closing bracket for the hard interrupt mode
        if (
            args.mode == "subsequent_interrupt"
            or args.mode == "subsequent_intervene_hard"
        ):
            updated_reasoning_trace = close_reasoning_trace(
                args.model_name, updated_reasoning_trace, force_answer=True
            )
        if args.mode == "subsequent_interrupt_extreme":
            updated_reasoning_trace = close_reasoning_trace(
                args.model_name, updated_reasoning_trace, force_answer=True
            )
            if args.task == "code":
                updated_reasoning_trace += "\n```"
            elif args.task == "math":
                updated_reasoning_trace += "\n\\boxed{"
        return formatted_prompt + updated_reasoning_trace

    elif args.interrupt_role == "user":
        truncated_reasoning_trace = close_reasoning_trace(
            args.model_name, truncated_reasoning_trace, force_answer=False
        )
        user_turn = format_user_turn(args.model_name, update_instructions)
        if args.mode == "subsequent_intervene_hard":
            user_turn += open_close_reasoning_trace(
                args.model_name
            )  # in the second turn, don't allow the model to think
        return formatted_prompt + truncated_reasoning_trace + user_turn
