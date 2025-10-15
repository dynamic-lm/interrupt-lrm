import os
from datasets import load_dataset, concatenate_datasets

GSM8K_PATH = "openai/gsm8k"
GSM8K_SAMPLE_SIZE = 500
MATH500_PATH = "HuggingFaceH4/MATH-500"
AIME2024_PATH = "math-ai/aime24"
AIME2025_PATH = "math-ai/aime25"

SEED = 42
OUTPUT_PATH = "../data/math.jsonl"


def main():
    start_id = 0

    all_datasets = []
    print(f"Loading GSM8K dataset...")
    gsm8k_dataset = load_dataset(GSM8K_PATH, name="main", split="test")
    gsm8k_dataset = gsm8k_dataset.shuffle(seed=SEED).select(range(GSM8K_SAMPLE_SIZE))
    gsm8k_dataset = gsm8k_dataset.rename_column("question", "problem")
    gsm8k_dataset = gsm8k_dataset.add_column("source", ["gsm8k"] * len(gsm8k_dataset))
    gsm8k_dataset = gsm8k_dataset.add_column(
        "id", [start_id + i for i in range(len(gsm8k_dataset))]
    )
    start_id += len(gsm8k_dataset)

    print(f"Loading MATH500 dataset...")
    math500_dataset = load_dataset(MATH500_PATH, split="test")
    math500_dataset = math500_dataset.select_columns(["problem", "answer"])
    math500_dataset = math500_dataset.add_column(
        "source", ["math500"] * len(math500_dataset)
    )
    math500_dataset = math500_dataset.add_column(
        "id", [start_id + i for i in range(len(math500_dataset))]
    )
    start_id += len(math500_dataset)

    all_datasets.append(gsm8k_dataset)
    all_datasets.append(math500_dataset)

    print(f"Loading AIME2024 dataset...")
    aime2024_dataset = load_dataset(AIME2024_PATH, split="test")
    aime2024_dataset = aime2024_dataset.rename_column("solution", "answer")
    aime2024_dataset = aime2024_dataset.select_columns(["problem", "answer"])
    aime2024_dataset = aime2024_dataset.add_column(
        "source", ["aime2024"] * len(aime2024_dataset)
    )
    aime2024_dataset = aime2024_dataset.add_column(
        "id", [start_id + i for i in range(len(aime2024_dataset))]
    )
    start_id += len(aime2024_dataset)
    all_datasets.append(aime2024_dataset)

    print(f"Loading AIME2025 dataset...")
    aime2025_dataset = load_dataset(AIME2025_PATH, split="test")
    aime2025_dataset = aime2025_dataset.select_columns(["problem", "answer"])
    aime2025_dataset = aime2025_dataset.add_column(
        "source", ["aime2025"] * len(aime2025_dataset)
    )
    aime2025_dataset = aime2025_dataset.add_column(
        "id", [start_id + i for i in range(len(aime2025_dataset))]
    )
    start_id += len(aime2025_dataset)
    all_datasets.append(aime2025_dataset)

    dataset = concatenate_datasets(all_datasets)
    print(f"Total number of problems: {len(dataset)}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset.to_json(OUTPUT_PATH, orient="records", lines=True)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
