"""
evaluate.py
End-to-end evaluation of the Hinglish NLP pipeline.

Pipeline:
  Raw text → HinglishTokenizer → HinglishNormalizer
           → LID CRF → POS CRF

Outputs:
  • Classification report (Precision, Recall, F1) for LID
  • Classification report (Precision, Recall, F1) for POS
  • 5 qualitative error examples for manual inspection
"""

import os, sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report

# ── Make src importable ────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.features.tokenizer  import HinglishTokenizer
from src.features.normalizer import HinglishNormalizer
from src.models.lid_crf      import load_sentences as load_lid_sents
from src.models.lid_crf      import sent2features  as lid_sent2features
from src.models.pos_crf      import load_pos_sentences, sent2pos_features

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_LID_CSV = os.path.join(ROOT, "data", "raw", "test.csv")
TEST_POS_CSV = os.path.join(ROOT, "data", "raw", "test_pos.csv")
LID_MODEL    = os.path.join(ROOT, "src", "models", "lid_model.pkl")
POS_MODEL    = os.path.join(ROOT, "src", "models", "pos_model.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def qualitative_errors(
    sentences:   list[list[str]],
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
    task_name:   str,
    n: int = 5,
):
    """
    Print up to *n* sentences that contain at least one prediction error,
    showing per-token (true, predicted) pairs to aid manual inspection.
    """
    print_section(f"Qualitative Error Analysis — {task_name}")
    shown = 0
    for sent, true, pred in zip(sentences, true_labels, pred_labels):
        if true == pred:
            continue
        print(f"\nSentence: {' '.join(sent)}")
        print(f"  {'Token':<20} {'True':<12} {'Predicted':<12} {'Match?'}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*6}")
        for tok, t, p in zip(sent, true, pred):
            match = "✓" if t == p else "✗"
            print(f"  {tok:<20} {t:<12} {p:<12} {match}")
        shown += 1
        if shown >= n:
            break

    if shown == 0:
        print("  No errors found — all predictions matched the ground truth!")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    # ── Load models ──────────────────────────────────────────────────────────
    print("[EVAL] Loading models …")
    lid_crf = joblib.load(LID_MODEL)
    pos_crf = joblib.load(POS_MODEL)
    print("  LID model loaded ✓")
    print("  POS model loaded ✓")

    # ── Instantiate pipeline components ──────────────────────────────────────
    tokenizer  = HinglishTokenizer()
    normalizer = HinglishNormalizer(spell_check=False)  # Disabled to avoid Hindi→English false corrections

    # =========================================================================
    # LID EVALUATION
    # =========================================================================
    print_section("Language Identification (LID) Evaluation")

    lid_sents, lid_true = load_lid_sents(TEST_LID_CSV)
    print(f"  Test sentences : {len(lid_sents):,}")
    print(f"  Test tokens    : {sum(len(s) for s in lid_sents):,}")

    # Run pipeline: normalize tokens before feature extraction
    lid_sents_norm = [normalizer.normalize_sentence(s) for s in lid_sents]
    X_lid = [lid_sent2features(s) for s in lid_sents_norm]
    lid_pred = lid_crf.predict(X_lid)

    # Flatten for sklearn metrics
    y_true_flat = [lab for sent_labs in lid_true for lab in sent_labs]
    y_pred_flat = [lab for sent_labs in lid_pred  for lab in sent_labs]

    print("\n" + classification_report(y_true_flat, y_pred_flat, digits=4, zero_division=0))

    qualitative_errors(lid_sents, lid_true, lid_pred, task_name="LID")

    # =========================================================================
    # POS EVALUATION
    # =========================================================================
    print_section("Part-of-Speech (POS) Tagging Evaluation")

    pos_sents, pos_true = load_pos_sentences(TEST_POS_CSV)
    print(f"  Test sentences : {len(pos_sents):,}")
    print(f"  Test tokens    : {sum(len(s) for s in pos_sents):,}")

    # Normalize → LID predict → POS predict
    pos_sents_norm  = [normalizer.normalize_sentence(s) for s in pos_sents]
    X_lid_pos       = [lid_sent2features(s) for s in pos_sents_norm]
    lid_pred_for_pos = lid_crf.predict(X_lid_pos)

    X_pos = [
        sent2pos_features(s, lid_tags)
        for s, lid_tags in zip(pos_sents_norm, lid_pred_for_pos)
    ]
    pos_pred = pos_crf.predict(X_pos)

    y_pos_true_flat = [lab for sent_labs in pos_true for lab in sent_labs]
    y_pos_pred_flat = [lab for sent_labs in pos_pred  for lab in sent_labs]

    print("\n" + classification_report(y_pos_true_flat, y_pos_pred_flat, digits=4, zero_division=0))

    qualitative_errors(pos_sents, pos_true, pos_pred, task_name="POS")

    print_section("Evaluation Complete")


if __name__ == "__main__":
    evaluate()
