import argparse
import os
import pandas as pd
from utils import score_response_with_timeout
from tqdm import tqdm

tasks = ["gsm8k", "math500", "aime2024", "aime2025"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--add_math_block", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_data = pd.read_json(args.input_file, lines=True)

    accuracy_dict = {
        "gsm8k": {"correct": 0, "total": 0},
        "math500": {"correct": 0, "total": 0},
        "aime2024": {"correct": 0, "total": 0},
        "aime2025": {"correct": 0, "total": 0},
    }

    # iterate over the merged data for each row
    acc_list = []
    for index, row in tqdm(prediction_data.iterrows(), total=len(prediction_data)):
        task = row["source"]
        ground_truth = row["answer"]
        prediction = row["output"][-1]
        if args.add_math_block:
            prediction = "\\boxed{" + prediction
        accuracy = score_response_with_timeout(task, ground_truth, prediction)
        accuracy_dict[task]["correct"] += int(accuracy)
        accuracy_dict[task]["total"] += 1
        acc_list.append(int(accuracy))

    # show the accuracy for each task
    for task in tasks:
        accuracy = accuracy_dict[task]["correct"] / accuracy_dict[task]["total"]
        print(
            f"{task}: {accuracy_dict[task]['correct']}/{accuracy_dict[task]['total']} ({accuracy:.2%})"
        )

    # show the overall accuracy
    overall_accuracy = sum(accuracy_dict[task]["correct"] for task in tasks) / sum(
        accuracy_dict[task]["total"] for task in tasks
    )

    print(f"Overall accuracy: {overall_accuracy:.2%}")

    result_df = pd.DataFrame({"id": prediction_data["id"], "pass@1": acc_list})

    result_df.to_json(args.output_file, orient="records", lines=True)
