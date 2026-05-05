"""
evaluate.py  —  Full accuracy report for the Hinglish NLP Pipeline (Approach 2)

Evaluates both sub-systems on the held-out TEST sets:
  1. CodeswitchSegmenter  (Language ID: HI / EN / UNI)
  2. HMMPosTagger         (POS tagging on Hinglish tokens)

Metrics reported
  • Overall accuracy
  • Per-class Precision / Recall / F1
  • Confusion matrix (top confusions)
  • OOV (out-of-vocabulary) breakdown
"""

import csv
import os
import sys
from collections import defaultdict
from tqdm import tqdm

# ── make project root importable ─────────────────────────────────────────────
# This file sits at hinglish/tools/evaluate.py → root is two dirs up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hinglish.statistical_models.hmm_pos_tagger import HMMPosTagger
from hinglish.statistical_models.lid_model import LIDModel

PUNC_SET = set(".,!?;:()[]{}") | {"--", "..."}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sentences(path: str) -> list[list[tuple[str, str]]]:
    """Read token/label CSV into a list of sentences (each = list of (tok, label))."""
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


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def per_class_metrics(y_true, y_pred, classes):
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in counts:
            if t == p:
                counts[t]["tp"] += 1
            else:
                counts[t]["fn"] += 1
        if p in counts and p != t:
            counts[p]["fp"] += 1
    rows = []
    for c in sorted(classes):
        d = counts[c]
        pr, rc, f1 = precision_recall_f1(d["tp"], d["fp"], d["fn"])
        support = d["tp"] + d["fn"]
        rows.append((c, pr, rc, f1, support))
    return rows


def top_confusions(y_true, y_pred, n=10):
    conf = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf[(t, p)] += 1
    return sorted(conf.items(), key=lambda x: -x[1])[:n]


# ── LID evaluation ────────────────────────────────────────────────────────────

def evaluate_lid():
    print("\n" + "=" * 65)
    print("  LANGUAGE IDENTIFICATION  (Trained Statistical LIDModel)")
    print("=" * 65)

    model = LIDModel()
    model.load()
    sentences = load_sentences("data/raw/hinglish_test_lid.csv")

    y_true, y_pred = [], []
    for sent in tqdm(sentences, desc="LID eval"):
        tokens = [tok for tok, _ in sent]
        truth  = [lbl for _, lbl in sent]   # already EN / HI / UNI
        preds  = model.predict(tokens)
        y_true.extend(truth)
        y_pred.extend(preds)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    print(f"\nOverall Accuracy : {correct/total:.2%}  ({correct}/{total} tokens)")

    classes = sorted(set(y_true) | set(y_pred))
    rows = per_class_metrics(y_true, y_pred, classes)

    print(f"\n{'Label':8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 52)
    for c, pr, rc, f1, sup in rows:
        print(f"{c:8} {pr:10.2%} {rc:10.2%} {f1:10.2%} {sup:10}")

    print(f"\nTop confusions (true → predicted):")
    for (t, p), cnt in top_confusions(y_true, y_pred, n=8):
        print(f"  {t:6} → {p:6}  ×{cnt}")

    return correct / total


# ── POS evaluation ────────────────────────────────────────────────────────────

def evaluate_pos():
    print("\n" + "=" * 65)
    print("  POS TAGGING  (HMM Viterbi Tagger)")
    print("=" * 65)

    tagger = HMMPosTagger()
    tagger.load()
    if not tagger.all_tags:
        print("ERROR: HMM model not loaded.")
        return 0.0

    sentences = load_sentences("data/raw/bis_test_pos.csv")

    y_true, y_pred = [], []
    oov_true, oov_pred = [], []   # tokens unseen during training
    seen_words = tagger.seen_words

    for sent in tqdm(sentences, desc="POS eval"):
        tokens    = [tok for tok, _ in sent]
        truth     = []
        for tok, lbl in sent:
            # normalise punctuation ground truth
            if tok in PUNC_SET:
                truth.append("PUNC")
            else:
                truth.append(lbl)

        preds = [tag for _, tag in tagger.viterbi_decode(tokens)]

        for tok, t, p in zip(tokens, truth, preds):
            y_true.append(t)
            y_pred.append(p)
            if tok.lower() not in seen_words:
                oov_true.append(t)
                oov_pred.append(p)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    print(f"\nOverall Accuracy : {correct/total:.2%}  ({correct}/{total} tokens)")

    oov_correct = sum(t == p for t, p in zip(oov_true, oov_pred))
    oov_total   = len(oov_true)
    if oov_total:
        print(f"OOV  Accuracy    : {oov_correct/oov_total:.2%}  ({oov_correct}/{oov_total} OOV tokens)")
    iv_correct  = correct - oov_correct
    iv_total    = total - oov_total
    if iv_total:
        print(f"In-Vocab Accuracy: {iv_correct/iv_total:.2%}  ({iv_correct}/{iv_total} in-vocab tokens)")

    classes = sorted(set(y_true) | set(y_pred))
    rows = per_class_metrics(y_true, y_pred, classes)

    print(f"\n{'Tag':10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    # sort by support descending so the most common tags appear first
    for c, pr, rc, f1, sup in sorted(rows, key=lambda x: -x[4]):
        if sup == 0 and pr == 0 and rc == 0:
            continue
        print(f"{c:10} {pr:10.2%} {rc:10.2%} {f1:10.2%} {sup:10}")

    print(f"\nTop confusions (true → predicted):")
    for (t, p), cnt in top_confusions(y_true, y_pred, n=10):
        print(f"  {t:8} → {p:8}  ×{cnt}")

    return correct / total


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  HINGLISH NLP PIPELINE — ACCURACY REPORT")
    print("  Test set: data/raw/bis_test_*.csv  (BIS/IIIT tagset, same-distribution)")
    print("=" * 65)

    lid_acc = evaluate_lid()
    pos_acc = evaluate_pos()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Language ID Accuracy  :  {lid_acc:.2%}")
    print(f"  POS Tagging Accuracy  :  {pos_acc:.2%}")
    print("=" * 65 + "\n")
