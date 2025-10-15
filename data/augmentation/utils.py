import json
import argparse
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_api_base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_data(input_file):
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def save_data(data, output_path):
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def generate(entry, args, json_format=False):
    completion_kwargs = {
        "model": args.model_name,
        "messages": entry["messages"],
        "temperature": args.temperature,
    }
    if args.reasoning_effort:
        completion_kwargs["reasoning_effort"] = args.reasoning_effort
    if json_format:
        completion_kwargs["response_format"] = {"type": "json_object"}
    if args.model_api_base:
        completion_kwargs["api_base"] = args.model_api_base
    max_retries = args.max_retries
    while max_retries > 0:
        try:
            response = completion(**completion_kwargs)
            if json_format:
                return json.loads(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating for entry {entry['id']}: {e}")
            max_retries -= 1
    if json_format:
        return dict()
    return None


def run_threaded(data, func, args, json_format=False):
    outputs = [None] * len(data)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_index = {
            executor.submit(func, entry, args, json_format): i
            for i, entry in enumerate(data)
        }

        with tqdm(total=len(data), desc="Processing data") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except Exception as e:
                    print(f"Error processing entry {index}: {e}")
                    outputs[index] = None
                pbar.update(1)
    return outputs
