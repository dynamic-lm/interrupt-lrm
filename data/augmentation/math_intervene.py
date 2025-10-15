from dotenv import load_dotenv

load_dotenv()

from prompts import (
    INTERVENE_MATH_SYS_PROMPT as SYS_PROMPT,
    INTERVENE_MATH_PROMPT as PROMPT_TEMPLATE,
    INTERVENE_MATH_UPDATE_SYS_PROMPT as UPDATE_SYS_PROMPT,
    INTERVENE_MATH_UPDATE_PROMPT as UPDATE_PROMPT_TEMPLATE,
)

COL_NAMES = [
    "id",
    "source",
    "original_problem",
    "original_answer",
    "revised_problem",
    "update",
]

from utils import (
    get_args,
    load_data,
    generate,
    run_threaded,
    save_data,
)


def main():
    args = get_args()
    data = load_data(args.input_file)

    # generate the revised problem
    for i in range(len(data)):
        data[i]["messages"] = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(problem=data[i]["problem"]),
            },
        ]
    outputs_1 = run_threaded(data, generate, args, json_format=False)

    # generate the update
    for i in range(len(data)):
        data[i]["messages"] = [
            {"role": "system", "content": UPDATE_SYS_PROMPT},
            {
                "role": "user",
                "content": UPDATE_PROMPT_TEMPLATE.format(
                    problem_a=outputs_1[i], problem_b=data[i]["problem"]
                ),
            },
        ]
    outputs_2 = run_threaded(data, generate, args, json_format=False)

    for i in range(len(data)):
        data[i]["original_problem"] = data[i]["problem"]
        data[i]["original_answer"] = data[i]["answer"]
        data[i]["revised_problem"] = outputs_1[
            i
        ]  # change the question content (updated problem)
        update = outputs_2[i]
        data[i]["update"] = update
        data[i] = {col: data[i][col] for col in COL_NAMES}

    save_data(data, args.output_path)


if __name__ == "__main__":
    main()
