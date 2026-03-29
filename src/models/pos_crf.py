"""
pos_crf.py
Trains a CRF model for Part-of-Speech (POS) tagging of Hinglish text.

Key design point: the predicted Language ID tag is used as an *input feature*
to the POS CRF, simulating a pipeline where LID runs before POS tagging.
"""

import os
import string
import joblib
import pandas as pd
import sklearn_crfsuite

# ── Sibling module import ────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.lid_crf import load_sentences as lid_load_sentences
from src.models.lid_crf import sent2features  as lid_sent2features

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_POS = os.path.join(ROOT, "data", "raw", "train_pos.csv")   # POS-annotated split
LID_MODEL = os.path.join(ROOT, "src", "models", "lid_model.pkl")
MODEL_OUT = os.path.join(ROOT, "src", "models", "pos_model.pkl")


# ---------------------------------------------------------------------------
# Feature extraction (POS-specific, augmented with LID predictions)
# ---------------------------------------------------------------------------

def _pos_features(
    sent:      list[str],
    lid_tags:  list[str],
    i:         int,
) -> dict:
    """
    Feature function for POS tagging.
    Includes all LID features PLUS the predicted language tag
    (and context language tags) as additional features.
    """
    # Start with the LID features for this position
    feats = lid_sent2features(sent)[i].copy()

    # Append Language ID tags as categorical features
    feats["lid_tag"] = lid_tags[i]
    if i > 0:
        feats["lid_tag-1"] = lid_tags[i - 1]
    if i < len(sent) - 1:
        feats["lid_tag+1"] = lid_tags[i + 1]

    return feats


def sent2pos_features(sent: list[str], lid_tags: list[str]) -> list[dict]:
    return [_pos_features(sent, lid_tags, i) for i in range(len(sent))]


# ---------------------------------------------------------------------------
# Data loading — POS CSV must have columns: token, label (POS tag)
# ---------------------------------------------------------------------------

def load_pos_sentences(csv_path: str):
    """Load POS-annotated CSV (same flat format as LID CSV)."""
    df = pd.read_csv(csv_path)
    sentences, labels = [], []
    cur_toks, cur_labs = [], []
    for _, row in df.iterrows():
        token = str(row["token"])
        label = str(row["label"])
        if token == "" or (isinstance(row.get("token"), float)):
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
# Training
# ---------------------------------------------------------------------------

def train(
    train_pos_csv: str = TRAIN_POS,
    lid_model_path: str = LID_MODEL,
    model_out: str = MODEL_OUT,
):
    print(f"[POS] Loading LID model from: {lid_model_path}")
    lid_crf = joblib.load(lid_model_path)

    print(f"[POS] Loading POS training data from: {train_pos_csv}")
    sents, pos_labels = load_pos_sentences(train_pos_csv)
    print(f"  {len(sents):,} sentences | {sum(len(s) for s in sents):,} tokens")

    print("[POS] Predicting LID tags for feature augmentation …")
    X_lid = [lid_sent2features(s) for s in sents]
    lid_preds = lid_crf.predict(X_lid)     # list[list[str]]

    print("[POS] Building POS feature vectors …")
    X_train = [sent2pos_features(s, lid_tags) for s, lid_tags in zip(sents, lid_preds)]
    y_train = pos_labels

    print("[POS] Training CRF …")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    joblib.dump(crf, model_out)
    print(f"[POS] Model saved → {model_out}")
    return crf


if __name__ == "__main__":
    train()
