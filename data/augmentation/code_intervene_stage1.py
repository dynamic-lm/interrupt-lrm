import random
from typing import List
from prompts import INTERVENE_CODE_PROMPT as PROMPT_TEMPLATE
from utils import (
    get_args,
    load_data,
    generate,
    run_threaded,
    save_data,
)

from dotenv import load_dotenv

load_dotenv()

SYNONYMS: List[str] = [
    "Kindly review the test cases and specifications provided below.",
    "Refer to the test cases and specifications outlined here.",
    "The following are the test cases and specifications for your reference.",
    "Below are the test cases and specifications to consider.",
    "Please take a look at the test cases and specifications listed below.",
    "Here are the test cases and specifications you should review.",
    "For your reference, see the test cases and specifications below.",
    "Find the test cases and specifications detailed here.",
    "Please consult the following test cases and specifications.",
    "Provided below are the test cases and specifications.",
    "The test cases and specifications are presented as follows.",
    "Kindly check the test cases and specifications outlined below.",
    "You can review the test cases and specifications listed here.",
    "Please examine the test cases and specifications below.",
    "See the details of the test cases and specifications provided.",
    "Below, you'll find the test cases and specifications.",
    "Review the following test cases and specifications carefully.",
    "Here are the specifications and test cases for your consideration.",
    "The test cases and specifications are included below.",
    "Please go through the test cases and specifications listed here.",
]


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
        data[i]["original_problem"] = data[i]["problem"]
        data[i]["revised_problem"] = outputs[i][
            "main_specifications"
        ]  # change the problem to the partial problem
        update_opening = random.choice(SYNONYMS)
        update = f"{update_opening}\n{outputs[i]['additional_specifications']}\n{outputs[i]['test_cases']}"
        data[i]["update"] = update
        del data[i]["messages"]

    save_data(data, args.output_path)


if __name__ == "__main__":
    main()
