from inference_utils import (
    load_data,
    save_data,
    load_tokenizer_and_llm,
    run_initial_round,
    run_subsequent_intervene,
)
from parser_utils import get_args


def main(args):
    data, custom_prompts = load_data(args)
    tokenizer, llm = load_tokenizer_and_llm(args)

    if args.mode == "initial":
        data = run_initial_round(args, data, custom_prompts, tokenizer, llm)
    elif args.mode in [
        "subsequent_interrupt_hard",
        "subsequent_interrupt_soft",
        "subsequent_interrupt_extreme",
    ]:
        data = run_subsequent_intervene(args, data, custom_prompts, tokenizer, llm)
    elif args.mode in ["subsequent_interrupt_update"]:
        data = run_subsequent_intervene(args, data, custom_prompts, tokenizer, llm)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    save_data(args, data)


if __name__ == "__main__":
    args = get_args()
    main(args)
