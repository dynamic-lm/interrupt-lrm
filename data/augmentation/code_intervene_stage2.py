from prompts import INTERVENE_NECESSARY_CODE_PROMPT_PROBLEM_BREAKDOWN as PROMPT_TEMPLATE
from utils import (
    get_args,
    load_data,
    generate,
    run_threaded,
    save_data,
)

from dotenv import load_dotenv

load_dotenv()


def main():
    args = get_args()
    data = load_data(args.input_file)
    for i in range(len(data)):
        data[i]["messages"] = [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(problem=data[i]["problem"]),
            }
        ]
    outputs = run_threaded(data, generate, args, json_format=True)

    for i in range(len(data)):
        data[i]["revised_problem"] = outputs[i][
            "augmented_problem"
        ]  # change the problem to the partial problem
        data[i]["update"] = outputs[i]["problem_correction"] + data[i].get("update", "")
        del data[i]["messages"]

    save_data(data, args.output_path)


if __name__ == "__main__":
    main()
