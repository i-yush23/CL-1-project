"""
lid_crf.py
Trains a CRF model for token-level Language Identification (LID)
on the LINCE Hinglish dataset and saves it to src/models/lid_model.pkl.

Labels in LINCE LID task:
  en   – English
  hi   – Hindi (Romanized)
  rest – named entities, foreign words, universal tokens, mixed
"""

import os, sys
import string
import joblib
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_CSV = os.path.join(ROOT, "data", "raw", "train.csv")
MODEL_OUT = os.path.join(ROOT, "src", "models", "lid_model.pkl")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _word_features(sent: list[str], i: int) -> dict:
    """
    Extract rich character- and context-level features for token at index i.
    sent : list of raw string tokens for one sentence.
    """
    word = sent[i]
    w    = word.lower()

    feats = {
        # Identity
        "word":         word,
        "word.lower":   w,
        "word.len":     len(word),

        # Shape
        "is_upper":     word.isupper(),
        "is_title":     word.istitle(),
        "is_digit":     word.isdigit(),
        "has_punct":    any(c in string.punctuation for c in word),
        "has_digit":    any(c.isdigit() for c in word),
        "is_alpha":     word.isalpha(),

        # Prefix / suffix
        "prefix1":      w[:1],
        "prefix2":      w[:2],
        "prefix3":      w[:3],
        "suffix1":      w[-1:],
        "suffix2":      w[-2:],
        "suffix3":      w[-3:],

        # Position
        "is_first":     i == 0,
        "is_last":      i == len(sent) - 1,
    }

    # Previous word context
    if i > 0:
        pw = sent[i - 1].lower()
        feats.update({
            "word-1":        sent[i - 1],
            "word-1.lower":  pw,
            "word-1.suffix2": pw[-2:],
            "word-1.is_title": sent[i - 1].istitle(),
        })
    else:
        feats["BOS"] = True   # Beginning of sentence

    # Next word context
    if i < len(sent) - 1:
        nw = sent[i + 1].lower()
        feats.update({
            "word+1":        sent[i + 1],
            "word+1.lower":  nw,
            "word+1.suffix2": nw[-2:],
            "word+1.is_title": sent[i + 1].istitle(),
        })
    else:
        feats["EOS"] = True   # End of sentence

    return feats


def sent2features(sent: list[str]) -> list[dict]:
    return [_word_features(sent, i) for i in range(len(sent))]

def sent2labels(labels: list[str]) -> list[str]:
    return labels


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentences(csv_path: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Reconstruct sentence-level token/label sequences from the flat CSV.
    The CSV has no sentence IDs, so we heuristically group by split
    (each contiguous block with token=='.' or end-of-group is treated as
    a sentence boundary – LINCE stores sequences separated by empty rows).
    """
    df = pd.read_csv(csv_path)

    sentences, labels = [], []
    cur_toks, cur_labs = [], []

    for _, row in df.iterrows():
        token = str(row["token"])
        label = str(row["label"])

        if token == "" or (isinstance(row.get("token"), float)):
            # empty row = sentence boundary
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

def train(train_csv: str = TRAIN_CSV, model_out: str = MODEL_OUT):
    print(f"[LID] Loading training data from: {train_csv}")
    sents, labs = load_sentences(train_csv)
    print(f"  {len(sents):,} sentences | {sum(len(s) for s in sents):,} tokens")

    X_train = [sent2features(s) for s in sents]
    y_train = [sent2labels(l)   for l in labs]

    print("[LID] Training CRF …")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    joblib.dump(crf, model_out)
    print(f"[LID] Model saved → {model_out}")
    return crf


if __name__ == "__main__":
    train()
