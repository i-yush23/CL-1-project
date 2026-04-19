"""
make_hinglish_lid.py
Extracts HI/EN LID labels from hinglish_crf_train_data.jsonl and
writes a proper train/test CSV split for the statistical LID model.

Outputs:
  data/raw/hinglish_train_lid.csv   (80%)
  data/raw/hinglish_test_lid.csv    (20%)
"""

import csv
import json
import os
import random
import sys
import string

JSONL     = "hinglish_crf_train_data.jsonl"
OUT_TRAIN = "data/raw/hinglish_train_lid.csv"
OUT_TEST  = "data/raw/hinglish_test_lid.csv"
SEED      = 42
RATIO     = 0.80

PUNC = set(string.punctuation) | {"...", "--"}

def token_label(tok, lbl):
    """Normalise label: non-alpha tokens -> UNI."""
    clean = tok.replace("'", "")
    if not clean.isalpha():
        return "UNI"
    return lbl  # HI or EN

def write_csv(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "label"])
        for i, sent in enumerate(sentences):
            for tok, lbl in sent:
                w.writerow([tok, lbl])
            if i < len(sentences) - 1:
                w.writerow(["", ""])

# ── load sentences ─────────────────────────────────────────────────────────
sentences = []
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        tokens = obj["tokens"]
        labels = obj["labels"]
        sent = [(tok, token_label(tok, lbl))
                for tok, lbl in zip(tokens, labels)]
        if sent:
            sentences.append(sent)

print(f"Loaded {len(sentences):,} sentences from {JSONL}")
total_tokens = sum(len(s) for s in sentences)
print(f"Total tokens: {total_tokens:,}")

from collections import Counter
tag_dist = Counter(lbl for s in sentences for _, lbl in s)
print("Label distribution:")
for lbl, cnt in sorted(tag_dist.items()):
    print(f"  {lbl:6} {cnt:8,}  ({cnt/total_tokens*100:.1f}%)")

# ── split ──────────────────────────────────────────────────────────────────
rng = random.Random(SEED)
rng.shuffle(sentences)
split = int(len(sentences) * RATIO)
train, test = sentences[:split], sentences[split:]

write_csv(train, OUT_TRAIN)
write_csv(test,  OUT_TEST)

print(f"\nWritten:")
print(f"  {OUT_TRAIN}: {len(train):,} sentences, {sum(len(s) for s in train):,} tokens")
print(f"  {OUT_TEST}:  {len(test):,} sentences, {sum(len(s) for s in test):,} tokens")
