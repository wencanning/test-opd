import argparse
import json
import random
from pathlib import Path

import pandas as pd


def build_sft_row(record: dict) -> dict | None:
    prompt = record.get("rendered_prompt")
    response = record.get("response") or record.get("trajectory_text")
    if not prompt or not response:
        return None

    return {
        "prompt": prompt,
        "response": response,
        "sample_id": record.get("sample_id"),
        "data_source": record.get("data_source"),
        "question": record.get("question"),
        "ground_truth": record.get("ground_truth"),
        "answer_correct": record.get("answer_correct"),
        "quality_score": record.get("quality_score"),
        "quality_pass": record.get("quality_pass"),
        "search_turn_count": record.get("search_turn_count"),
        "information_turn_count": record.get("information_turn_count"),
    }


def load_rows(input_jsonl: Path) -> list[dict]:
    rows = []
    with input_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = build_sft_row(record)
            if row is not None:
                rows.append(row)
    return rows


def split_rows(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)

    val_size = int(len(rows) * val_ratio)
    if val_size <= 0 and len(rows) > 1:
        val_size = 1
    if val_size >= len(rows):
        val_size = max(1, len(rows) - 1)

    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser(description="Convert accepted Search-R1 rollouts into masked SFT parquet files.")
    parser.add_argument("--input_jsonl", required=True, help="Path to accepted_rollouts.jsonl")
    parser.add_argument("--output_dir", required=True, help="Directory to write train/val parquet files")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_jsonl)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)

    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(output_dir / "val.parquet", index=False)

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
