import os
import re
import json
import pickle
import zlib
import base64
import argparse
import pandas as pd
from tqdm import tqdm

from compute_code_generation_metrics import codegen_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_predictions", type=str, required=True)
    parser.add_argument("--input_test_cases", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--add_code_block", action="store_true")
    return parser.parse_args()


args = get_args()


def parse_output(output, model_name):
    if "qwen" in model_name.lower():
        pattern = r"(.*)</think>(.*)"
    elif "gptoss" in model_name.lower():
        pattern = (
            r"(.*)<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>(.*)"
        )
    elif "mistral" in model_name.lower():
        pattern = r"(.*)\[/THINK\](.*)"
    else:
        raise ValueError(f"Model name {model_name} not supported")
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        output_thinking, output_answer = matches[-1]
        return output_thinking, output_answer
    return None, output


def parse_code(output):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def process_test_cases(test_cases):
    try:
        return json.loads(test_cases)
    except:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8"))))
        )


def main():
    with open(args.input_predictions, "r") as f:
        generations = [json.loads(line) for line in f]

    for gen in generations:
        final_output = gen["output"][-1]
        # Support force output mode
        if args.add_code_block:
            final_output = "```" + final_output
        # thinking_content, answer_content = parse_output(final_output, args.model_name)
        # gen['thinking'] = thinking_content.strip() if thinking_content else None
        # gen['answer'] = answer_content.strip() if answer_content else None
        gen["code"] = parse_code(final_output)

    generations_df = pd.DataFrame(generations)
    print(
        f"Number of generations with empty code: {generations_df['code'].isna().sum()}"
    )
    print(f"Number of generated samples: {len(generations_df)}")

    with open(args.input_test_cases, "r") as f:
        input = [json.loads(line) for line in f]
    input_df = pd.DataFrame(input)

    data_df = pd.merge(input_df, generations_df[["id", "code"]], on="id", how="inner")
    print(f"Number of samples after merging with source dataset: {len(data_df)}")
    print(data_df.columns)
    print(data_df.iloc[0])

    samples_list, generations_list = [], []
    for index in tqdm(range(len(data_df))):
        entry = data_df.iloc[index]
        public_test_cases = process_test_cases(entry["public_test_cases"])
        private_test_cases = process_test_cases(entry["private_test_cases"])
        samples_list.append(
            {
                "input_output": json.dumps(
                    {
                        "inputs": [
                            test["input"]
                            for test in public_test_cases + private_test_cases
                        ],
                        "outputs": [
                            test["output"]
                            for test in public_test_cases + private_test_cases
                        ],
                        "fn_name": json.loads(entry["metadata"]).get("func_name", None),
                    }
                )
            }
        )
        generations_list.append([entry["code"]] if entry["code"] else [""])

    test_results = codegen_metrics(samples_list, generations_list)
    assert len(test_results[0]["detail"]["pass@1"]) == len(data_df)
    pass_at_1 = []
    for idx in range(len(data_df)):
        pass_at_1.append(test_results[0]["detail"]["pass@1"][idx])
    data_df["pass@1"] = pass_at_1
    print(f"Pass@1: {data_df['pass@1'].mean()}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    data_df[["id", "pass@1"]].to_json(args.output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
