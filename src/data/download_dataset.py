"""
download_dataset.py
Downloads the LINCE Hinglish dataset (LID + POS) from Hugging Face,
saves train/test splits to data/raw/, and prints basic EDA stats.
"""

import os
import random
import pandas as pd
from datasets import load_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_and_save(task: str = "lid_hineng"):
    """Download a LINCE split and save as CSVs."""
    print(f"[INFO] Downloading LINCE task: {task} …")
    dataset = load_dataset("lince", task)

    rows = []
    for split_name, split_data in dataset.items():
        records = []
        for example in split_data:
            tokens = example["tokens"]
            labels = example["lid_labels"] if "lid_labels" in example else example["pos_labels"]
            for tok, lab in zip(tokens, labels):
                records.append({"token": tok, "label": lab, "split": split_name})
        df = pd.DataFrame(records)
        path = os.path.join(RAW_DIR, f"{split_name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} token rows → {path}")
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def run_eda(df: pd.DataFrame):
    """Print basic exploratory statistics about the dataset."""
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    total_tokens = len(df)
    print(f"\nTotal tokens         : {total_tokens:,}")

    label_counts = df["label"].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        pct = count / total_tokens * 100
        print(f"  {label:<15} {count:>6,}  ({pct:.1f}%)")

    # English vs Hindi ratio (LINCE uses: en, hi, rest=ne/fw/univ/mixed)
    en_count = label_counts.get("en", 0)
    hi_count = label_counts.get("hi", 0)
    if en_count + hi_count > 0:
        ratio = en_count / (en_count + hi_count)
        print(f"\nEnglish/(En+Hi) ratio: {ratio:.2%}")

    # Random sample of 5 token-label pairs
    print("\nRandom sample (5 tokens):")
    sample = df.sample(5, random_state=42)[["token", "label"]].values.tolist()
    for tok, lab in sample:
        print(f"  '{tok}' → {lab}")

    print("=" * 50)


if __name__ == "__main__":
    df = download_and_save(task="lid_hineng")
    run_eda(df)
    print("\n[DONE] Raw data saved to data/raw/")
