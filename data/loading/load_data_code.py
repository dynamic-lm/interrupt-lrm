import os
from datetime import datetime
from datasets import load_dataset

LIVECODEBENCH_PATH = "livecodebench/code_generation_lite"
LIVECODEBENCH_VERSION = "release_v6"
START_DATE = datetime.strptime("2024-10-01", "%Y-%m-%d")
END_DATE = datetime.strptime("2025-05-01", "%Y-%m-%d")

SEED = 42
OUTPUT_PATH = "../data/code/data.jsonl"


def main():
    start_id = 0

    print(f"Loading LiveCodeBench dataset...")
    livecodebench_dataset = load_dataset(
        LIVECODEBENCH_PATH,
        split="test",
        version_tag=LIVECODEBENCH_VERSION,
        trust_remote_code=True,
    )
    contest_dates = [
        datetime.fromisoformat(ex["contest_date"]) for ex in livecodebench_dataset
    ]
    livecodebench_dataset = livecodebench_dataset.remove_columns(["contest_date"])
    livecodebench_dataset = livecodebench_dataset.add_column(
        "contest_date", contest_dates
    )
    livecodebench_dataset = livecodebench_dataset.filter(
        lambda ex: ex["contest_date"] >= START_DATE if START_DATE is not None else True
    )
    livecodebench_dataset = livecodebench_dataset.filter(
        lambda ex: ex["contest_date"] <= END_DATE if END_DATE is not None else True
    )
    livecodebench_dataset = livecodebench_dataset.add_column(
        "id", [start_id + i for i in range(len(livecodebench_dataset))]
    )
    livecodebench_dataset = livecodebench_dataset.rename_column(
        "question_content", "problem"
    )

    print(f"Total number of problems: {len(livecodebench_dataset)}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    livecodebench_dataset.to_json(OUTPUT_PATH, orient="records", lines=True)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
