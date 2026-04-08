import os
import re
import string
import joblib
import pandas as pd
import sklearn_crfsuite
import sys

# ── Sibling module import ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# Import the updated feature extractor from LID module
from src.approach1.models.lid_crf import sent2features as lid_sent2features

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_POS = os.path.join(ROOT, "data", "raw", "train_pos.csv")   # POS-annotated split
LID_MODEL = os.path.join(ROOT, "src", "approach1", "models", "lid_model.pkl")
MODEL_OUT = os.path.join(ROOT, "src", "approach1", "models", "pos_model.pkl")

# ── Lexicons for POS Anchoring ──────────────────────────────────────────────
# Helps disambiguate common function words
HI_VERB_SUFFIX = ("na", "ne", "ta", "te", "ti", "raha", "rahe", "rahi", "unga", "oge")
HI_PSP = {"se", "ka", "ki", "ko", "mein", "par", "ne", "ke"} # Postpositions

# ---------------------------------------------------------------------------
# Helper: Text Normalization
# ---------------------------------------------------------------------------

def normalize_hinglish(word: str) -> str:
    """
    Reduces noise common in Hinglish social media text.
    Ensures 'haaaaai' and 'hai' are seen as the same token.
    """
    word = word.lower()
    word = re.sub(r'(.)\1{2,}', r'\1\1', word)
    return word

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
    Incorporate LID tags as strong categorical signals.
    """
    word = sent[i]
    w_norm = normalize_hinglish(word)
    
    # Start with the base LID features (prefixes, suffixes, vowel ratios, etc.)
    feats = lid_sent2features(sent)[i].copy()

    # --- 1. Morphological Features (Deeper context for Verbs/Nouns) ---
    feats.update({
        'w_norm':      w_norm,
        'suffix_4':    w_norm[-4:], # Important for transliterated Hindi verbs
        'is_hi_verb':  (lid_tags[i] == "HI" and w_norm.endswith(HI_VERB_SUFFIX)),
        'is_psp':      w_norm in HI_PSP,
    })

    # --- 2. Orthographic Features ---
    feats.update({
        'is_numeric':  word.isdigit(),
        'is_punct':    all(c in string.punctuation for c in word),
        'shape':       re.sub(r'[A-Z]', 'X', re.sub(r'[a-z]', 'x', word)),
    })

    # --- 3. Augmented Context & LID Transition Features ---
    # Knowing we just switched languages is a high-quality signal for POS boundaries
    feats["lid_tag"] = lid_tags[i]
    
    if i > 0:
        feats["prev_lid_tag"] = lid_tags[i - 1]
        feats["prev_w_norm"]  = normalize_hinglish(sent[i - 1])
        feats["is_switch"]    = (lid_tags[i] != lid_tags[i - 1])
    else:
        feats["BOS"] = True

    if i < len(sent) - 1:
        feats["next_lid_tag"] = lid_tags[i + 1]
        feats["next_w_norm"]  = normalize_hinglish(sent[i + 1])
    else:
        feats["EOS"] = True

    return feats


def sent2pos_features(sent: list[str], lid_tags: list[str]) -> list[dict]:
    return [_pos_features(sent, lid_tags, i) for i in range(len(sent))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pos_sentences(csv_path: str):
    """Load POS-annotated CSV."""
    df = pd.read_csv(csv_path)
    sentences, labels = [], []
    cur_toks, cur_labs = [], []
    
    for _, row in df.iterrows():
        token = str(row["token"])
        label = str(row["label"])
        
        if token == "" or token.lower() == "nan" or pd.isna(row["token"]):
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
    
    # Ensure LID predictions are generated using the same features as POS features
    print("[POS] Predicting LID tags for feature augmentation …")
    X_lid = [lid_sent2features(s) for s in sents]
    lid_preds = lid_crf.predict(X_lid)

    print("[POS] Building POS feature vectors …")
    X_train = [sent2pos_features(s, lid_tags) for s, lid_tags in zip(sents, lid_preds)]
    y_train = pos_labels

    print("[POS] Training CRF …")
    # c1 (L1) is kept high to force the model to pick only the most predictive
    # morphological and transition features.
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.25,
        c2=0.05,
        max_iterations=250, 
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(crf, model_out)
    print(f"[POS] Model saved → {model_out}")
    
    return crf

if __name__ == "__main__":
    train()