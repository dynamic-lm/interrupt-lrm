import argparse


def add_common_args(parser):
    # Input, Output, Parallelism, and Seed
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism per model instance",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--task",
        type=str,
        default="math",
        choices=["math", "code"],
        help="Task to run the model",
    )
    return parser


def add_model_args(parser):
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    return parser


def add_mode_args(parser):
    parser.add_argument(
        "--problem_field_name",
        type=str,
        default="problem",
        help="Field name for the problem in the input JSONL file",
    )
    parser.add_argument(
        "--starter_code_field_name",
        type=str,
        default="starter_code",
        help="Field name for the starter code in the input JSONL file (only for code generation task)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="initial",
        choices=[
            "initial",
            "subsequent_interrupt_hard",
            "subsequent_interrupt_update",
            "subsequent_interrupt_soft",
            "subsequent_interrupt_extreme",
        ],
        help="Mode to run the model",
    )
    parser.add_argument(
        "--interrupt_pos",
        type=float,
        help="Relative position of the thinking trace to intervene at",
    )
    parser.add_argument(
        "--interrupt_role",
        type=str,
        default="assistant",
        choices=["assistant", "user"],
        help="For soft interrupt and intervene modes, the role the update instructions should be formatted as",
    )
    parser.add_argument(
        "--custom_prompt_file",
        type=str,
        default=None,
        help="Path to the custom prompt file",
    )
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser = add_model_args(parser)
    parser = add_mode_args(parser)
    args = parser.parse_args()
    return args


def get_args_modal(*arglist):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser = add_model_args(parser)
    parser = add_mode_args(parser)
    args = parser.parse_args(args=arglist)
    return args
