"""
lid_crf.py
Trains a CRF model for token-level Language Identification (LID)
using the aggregated Hinglish JSONL dataset.
"""

import os
import json
import string
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
# Updated to match the output of your download script
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH   = os.path.join(ROOT, "hinglish_crf_train_data.jsonl") 
MODEL_OUT   = os.path.join(ROOT, "src", "models", "lid_model.pkl")

# Ensure the model directory exists
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ---------------------------------------------------------------------------
# Feature extraction (Linguistic/Orthographic focus)
# ---------------------------------------------------------------------------

def _word_features(sent: list[str], i: int) -> dict:
    word = sent[i]
    w = word.lower()

    feats = {
        "bias":         1.0,
        "word.lower":   w,
        "word.len":     len(word),
        "is_upper":     word.isupper(),
        "is_title":     word.istitle(),
        "is_digit":     word.isdigit(),
        "is_alpha":     word.isalpha(),
        "has_punct":    any(c in string.punctuation for c in word),
        
        # Morphological features (crucial for Romanized Hindi vs English)
        "prefix1":      w[:1],
        "prefix2":      w[:2],
        "prefix3":      w[:3],
        "suffix1":      w[-1:],
        "suffix2":      w[-2:],
        "suffix3":      w[-3:],
    }

    # Contextual features (Transition modeling)
    if i > 0:
        pw = sent[i - 1].lower()
        feats.update({
            "word-1.lower":  pw,
            "word-1.suffix2": pw[-2:],
            "word-1.istitle": sent[i - 1].istitle(),
        })
    else:
        feats["BOS"] = True

    if i < len(sent) - 1:
        nw = sent[i + 1].lower()
        feats.update({
            "word+1.lower":  nw,
            "word+1.suffix2": nw[-2:],
            "word+1.istitle": sent[i + 1].istitle(),
        })
    else:
        feats["EOS"] = True

    return feats

def sent2features(sent: list[str]) -> list[dict]:
    return [_word_features(sent, i) for i in range(len(sent))]

# ---------------------------------------------------------------------------
# Data loading (JSONL Optimized)
# ---------------------------------------------------------------------------

def load_data_from_jsonl(file_path: str):
    """
    Reads the JSONL file and returns token sequences and label sequences.
    """
    sentences, labels = [], []
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found: {file_path}")
        return [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Basic validation to ensure sequence lengths match
            if len(data['tokens']) == len(data['labels']):
                sentences.append(data['tokens'])
                labels.append(data['labels'])
                
    return sentences, labels

# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train(data_path: str = DATA_PATH, model_out: str = MODEL_OUT):
    print(f"[LID] Loading data from: {data_path}")
    sents, labs = load_data_from_jsonl(data_path)
    
    if not sents:
        return

    print(f"  Total Sequences: {len(sents):,}")
    print(f"  Total Tokens   : {sum(len(s) for s in sents):,}")

    # Split into train/test for a sanity check (80/20)
    split_idx = int(len(sents) * 0.8)
    
    X_train = [sent2features(s) for s in sents[:split_idx]]
    y_train = labs[:split_idx]
    
    X_test = [sent2features(s) for s in sents[split_idx:]]
    y_test = labs[split_idx:]

    print("[LID] Training CRF with L-BFGS...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,  # Elastic net regularization
        c2=0.1,
        max_iterations=150,
        all_possible_transitions=True,
    )
    
    crf.fit(X_train, y_train)

    # Internal Evaluation
    print("\n[LID] Evaluation Results:")
    y_pred = crf.predict(X_test)
    print(crf_metrics.flat_classification_report(y_test, y_pred, digits=3))

    # Save the model
    joblib.dump(crf, model_out)
    print(f"\n[LID] Model successfully saved to {model_out}")
    return crf

if __name__ == "__main__":
    train()