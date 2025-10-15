import random
from prompts import INTERVENE_CODE_PROMPT_STARTER_CODE as PROMPT_TEMPLATE
from utils import (
    get_args,
    load_data,
    generate,
    run_threaded,
    save_data,
)

COL_NAMES = [
    "id",
    "platform",
    "contest_id",
    "difficulty",
    "question_title",
    "original_starter_code",
    "original_problem",
    "public_test_cases",
    "private_test_cases",
    "metadata",
    "revised_starter_code",
    "revised_problem",
    "update",
]

from dotenv import load_dotenv

load_dotenv()


def main():
    args = get_args()
    data = load_data(args.input_file)
    curr_indices = []
    curr_data = []
    for i in range(len(data)):
        if data[i]["starter_code"] != "":
            # augment the starter code + starter code update for all easy cases and 70% of (medium + hard) cases
            if data[i]["difficulty"] == "easy":
                curr_data.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATE.format(
                                    starter_code=data[i]["starter_code"]
                                ),
                            }
                        ]
                    }
                )
                curr_indices.append(i)
            elif data[i]["difficulty"] == "medium" or data[i]["difficulty"] == "hard":
                if random.random() < 0.7:
                    curr_data.append(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": PROMPT_TEMPLATE.format(
                                        starter_code=data[i]["starter_code"]
                                    ),
                                }
                            ]
                        }
                    )
                    curr_indices.append(i)

    outputs = run_threaded(curr_data, generate, args, json_format=True)

    for i in range(len(curr_data)):
        curr_idx = curr_indices[i]
        data[curr_idx]["original_starter_code"] = data[curr_idx]["starter_code"]
        data[curr_idx]["revised_starter_code"] = outputs[i]["new_starter_code"]
        data[curr_idx]["update"] = outputs[i]["correction"] + data[curr_idx].get(
            "update", ""
        )

    for i in range(len(data)):
        data[i] = {k: data[i].get(k, "") for k in COL_NAMES}

    save_data(data, args.output_path)


if __name__ == "__main__":
    main()
