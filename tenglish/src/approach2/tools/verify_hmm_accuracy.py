"""
verify_hmm_accuracy.py
Evaluates the HMM POS tagger and CodeswitchSegmenter against a labelled
test CSV, printing per-class and overall accuracy metrics.

Expected CSV format (same as training):
    token,label
    Nuvvu,PRON
    chestunnav,V
    ,
    ...

Usage:
    python verify_hmm_accuracy.py --pos  data/raw/test_pos.csv
    python verify_hmm_accuracy.py --lid  data/raw/test_lid.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from approach2.statistical_models.code_switch_labeler import CodeswitchSegmenter
from approach2.statistical_models.hmm_pos_tagger      import HMMPosTagger


# ---------------------------------------------------------------------------
# Data loader: groups CSV rows into sentences
# ---------------------------------------------------------------------------
def load_sentences(csv_path: str) -> tuple[list[list[str]], list[list[str]]]:
    sentences, labels = [], []
    cur_toks, cur_labs = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row.get("token", "").strip()
            label = row.get("label", "").strip()
            if not token or not label:
                if cur_toks:
                    sentences.append(cur_toks)
                    labels.append(cur_labs)
                    cur_toks, cur_labs = [], []
            else:
                cur_toks.append(token)
                cur_labs.append(label)

    if cur_toks:
        sentences.append(cur_toks)
        labels.append(cur_labs)

    return sentences, labels


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------
def classification_report(y_true: list[str], y_pred: list[str]) -> None:
    classes = sorted(set(y_true) | set(y_pred))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    total_tp = total_fp = total_fn = 0
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup  = tp[cls] + fn[cls]
        total_tp += tp[cls]; total_fp += fp[cls]; total_fn += fn[cls]
        print(f"{cls:<12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {sup:>10}")

    overall_acc = sum(tp.values()) / len(y_true) if y_true else 0.0
    print(f"\nOverall accuracy: {overall_acc:.4f}  ({sum(tp.values())}/{len(y_true)} correct)")


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------
def evaluate_pos(csv_path: str) -> None:
    print(f"\n=== POS Evaluation: {csv_path} ===")
    sents, true_labels = load_sentences(csv_path)

    tagger = HMMPosTagger()
    tagger.load()

    y_true, y_pred = [], []
    errors_shown   = 0

    for sent, true in zip(sents, true_labels):
        pred_pairs = tagger.viterbi_decode(sent)
        pred = [p for _, p in pred_pairs]

        y_true.extend(true)
        y_pred.extend(pred)

        if true != pred and errors_shown < 3:
            print(f"\n[Error example] {' '.join(sent)}")
            print(f"  {'Token':<18} {'True':<10} {'Pred':<10} {'Match'}")
            for tok, t, p in zip(sent, true, pred):
                mark = "✓" if t == p else "✗"
                print(f"  {tok:<18} {t:<10} {p:<10} {mark}")
            errors_shown += 1

    classification_report(y_true, y_pred)


def evaluate_lid(csv_path: str) -> None:
    print(f"\n=== LID Evaluation: {csv_path} ===")
    sents, true_labels = load_sentences(csv_path)

    segmenter = CodeswitchSegmenter()

    y_true, y_pred = [], []
    errors_shown   = 0

    for sent, true in zip(sents, true_labels):
        pred = segmenter.segment(sent)
        y_true.extend(true)
        y_pred.extend(pred)

        if true != pred and errors_shown < 3:
            print(f"\n[Error example] {' '.join(sent)}")
            print(f"  {'Token':<18} {'True':<6} {'Pred':<6} {'Match'}")
            for tok, t, p in zip(sent, true, pred):
                mark = "✓" if t == p else "✗"
                print(f"  {tok:<18} {t:<6} {p:<6} {mark}")
            errors_shown += 1

    classification_report(y_true, y_pred)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tenglish HMM pipeline")
    parser.add_argument("--pos", metavar="CSV", help="POS-labelled test CSV")
    parser.add_argument("--lid", metavar="CSV", help="LID-labelled test CSV")
    args = parser.parse_args()

    if not args.pos and not args.lid:
        parser.print_help()
        sys.exit(0)

    if args.pos:
        evaluate_pos(args.pos)
    if args.lid:
        evaluate_lid(args.lid)
