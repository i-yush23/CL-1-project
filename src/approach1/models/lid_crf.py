import os
import json
import string
import joblib
import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH   = os.path.join(ROOT, "hinglish_crf_train_data.jsonl") 
MODEL_OUT   = os.path.join(ROOT, "src", "approach1", "models", "lid_model.pkl")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ── Lexicons ──────────────────────────────────────────────────────────────────
HI_LEXICON = {"yr", "yaar", "ni", "nahi", "na", "se", "ka", "ki", "ko", "kuch", "ho", "hona", "aaj", "baat", "iss", "baar", "tum"}
EN_LEXICON = {"to", "be", "honest", "the", "a", "is", "of", "please", "dont", "fuck", "up", "awesome", "statistical", "models", "tbh"}

# ---------------------------------------------------------------------------
# Helper: Linguistic Signal Extraction
# ---------------------------------------------------------------------------

def get_vowel_ratio(word):
    vowels = "aeiou"
    v_count = sum(1 for c in word.lower() if c in vowels)
    return v_count / len(word) if len(word) > 0 else 0

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _word_features(sent: list[str], i: int, feature_level: int = 4) -> dict:
    word = sent[i]
    # Normalize elongation: "haaaaaai" -> "hai"
    w = re.sub(r'(.)\1{2,}', r'\1\1', word.lower())

    # Level 0: Baseline
    feats = {
        "bias":           1.0,
        "word.lower":     w,
    }

    # Level 1: Basic string shapes
    if feature_level >= 1:
        feats.update({
            "word.len":       len(w),
            "is_upper":       word.isupper(),
            "is_digit":       word.isdigit(),
            "has_punct":      any(c in string.punctuation for c in word),
        })

    # Level 2: Morphology & Phonetics base
    if feature_level >= 2:
        feats.update({
            "v_ratio":        get_vowel_ratio(w),
            "p1": w[:1], "p2": w[:2], "p3": w[:3],
            "s1": w[-1:], "s2": w[-2:], "s3": w[-3:], "s4": w[-4:],
        })

    # Level 3: Lexicons & Advanced Phonetics
    if feature_level >= 3:
        feats.update({
            "is_hi_lex":      w in HI_LEXICON,
            "is_en_lex":      w in EN_LEXICON,
            "aspirated":      any(sub in w for sub in ["kh", "gh", "bh", "dh", "sh", "th"]),
        })

    # Level 4: Context
    if feature_level >= 4:
        if i > 0:
            feats.update({"prev_w": sent[i-1].lower(), "prev_is_hi": sent[i-1].lower() in HI_LEXICON})
        else:
            feats["BOS"] = True

        if i < len(sent) - 1:
            feats.update({"next_w": sent[i+1].lower(), "next_is_en": sent[i+1].lower() in EN_LEXICON})
        else:
            feats["EOS"] = True

    return feats

def sent2features(sent: list[str], feature_level: int = 4) -> list[dict]:
    return [_word_features(sent, i, feature_level) for i in range(len(sent))]


def load_data_from_jsonl(file_path: str):
    sentences, labels = [], []
    if not os.path.exists(file_path): return [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if len(data['tokens']) == len(data['labels']):
                sentences.append(data['tokens'])
                labels.append(data['labels'])
    return sentences, labels

def train(data_path: str = DATA_PATH, model_out: str = MODEL_OUT, feature_level: int = 4):
    sents, labs = load_data_from_jsonl(data_path)
    if not sents: return

    split_idx = int(len(sents) * 0.8)
    X_train = [sent2features(s, feature_level) for s in sents[:split_idx]]
    y_train = labs[:split_idx]
    X_test  = [sent2features(s, feature_level) for s in sents[split_idx:]]
    y_test  = labs[split_idx:]

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.25, c2=0.05, max_iterations=200, all_possible_transitions=True)
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    print(f"\n[LID] Results for Feature Level {feature_level}:\n", crf_metrics.flat_classification_report(y_test, y_pred, digits=3))
    joblib.dump(crf, model_out)
    return crf

if __name__ == "__main__":
    train()