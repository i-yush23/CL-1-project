"""
make_bis_test.py — carve out a BIS-tagged held-out test set from the
Hinglish training corpus using a reproducible 80/20 sentence split.

Outputs:
  data/raw/bis_test_pos.csv   — POS labels (BIS/IIIT tagset)
  data/raw/bis_test_lid.csv   — LID labels (HI / EN / UNI)
                                 inferred via CodeswitchSegmenter
"""
import csv
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hinglish.statistical_models.code_switch_labeler import CodeswitchSegmenter

SEED = 42
TRAIN_RATIO = 0.80

TRAIN_POS  = "data/raw/train_pos.csv"
TRAIN_LID  = "data/raw/train_lid.csv"
OUT_POS    = "data/raw/bis_test_pos.csv"
OUT_LID    = "data/raw/bis_test_lid.csv"

# ── load sentences ────────────────────────────────────────────────────────────
def load(path):
    sentences, cur = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tok, lbl = row["token"].strip(), row["label"].strip()
            if not tok or not lbl:
                if cur:
                    sentences.append(cur)
                    cur = []
            else:
                cur.append((tok, lbl))
    if cur:
        sentences.append(cur)
    return sentences


def write(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "label"])
        for i, sent in enumerate(sentences):
            for tok, lbl in sent:
                w.writerow([tok, lbl])
            if i < len(sentences) - 1:
                w.writerow(["", ""])   # sentence boundary


# ── split ─────────────────────────────────────────────────────────────────────
pos_sents = load(TRAIN_POS)
lid_sents = load(TRAIN_LID)

# Align on sentence count (POS corpus is slightly smaller)
n = min(len(pos_sents), len(lid_sents))
pos_sents = pos_sents[:n]
lid_sents = lid_sents[:n]

# Reproducible shuffle by index
rng = random.Random(SEED)
indices = list(range(n))
rng.shuffle(indices)

split = int(n * TRAIN_RATIO)
test_idx = sorted(indices[split:])

test_pos = [pos_sents[i] for i in test_idx]
test_lid = [lid_sents[i] for i in test_idx]

train_idx = sorted(indices[:split])
train_lid_sents = [lid_sents[i] for i in train_idx]

write(test_pos, OUT_POS)
write(test_lid, OUT_LID)
write(train_lid_sents, "data/raw/bis_train_lid.csv")

pos_tokens = sum(len(s) for s in test_pos)
lid_tokens = sum(len(s) for s in test_lid)
lid_train_tokens = sum(len(s) for s in train_lid_sents)
print(f"BIS splits written:")
print(f"  {OUT_POS}: {len(test_pos)} sentences, {pos_tokens} tokens  (POS test)")
print(f"  {OUT_LID}: {len(test_lid)} sentences, {lid_tokens} tokens  (LID test)")
print(f"  data/raw/bis_train_lid.csv: {len(train_lid_sents)} sentences, {lid_train_tokens} tokens  (LID train)")

# Tag distribution
from collections import Counter
pos_tags = Counter(lbl for s in test_pos for _, lbl in s)
print(f"\nTop POS tags in BIS test set:")
for tag, cnt in pos_tags.most_common(15):
    print(f"  {tag:10} {cnt:5}")
