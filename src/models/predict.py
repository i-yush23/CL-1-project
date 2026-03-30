import os
import sys
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.features.tokenizer import HinglishTokenizer
from src.features.normalizer import CodeMixedNormalizer
from src.models.lid_crf import sent2features as lid_sent2features
from src.models.pos_crf import sent2pos_features

LID_MODEL = os.path.join(ROOT, "src", "models", "lid_model.pkl")
POS_MODEL = os.path.join(ROOT, "src", "models", "pos_model.pkl")

def test_pipeline(text: str):
    print(f"\n--- Testing Pipeline ---")
    print(f"Input: {text}")
    
    # 1. Tokenize
    tokenizer = HinglishTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # 2. Normalize
    normalizer = CodeMixedNormalizer()
    norm_dicts = normalizer.process(tokens)
    
    # Flatten if normalization expanded an abbreviation into multiple words (e.g. 'tbh' -> 'to be honest')
    flat_orig = []
    flat_norm = []
    for orig_t, d in zip(tokens, norm_dicts):
        norm_str = d["norm"]
        split_norms = norm_str.split()
        if len(split_norms) > 1:
            for sn in split_norms:
                flat_orig.append(orig_t) # Keep orig placeholder for alignment visualization
                flat_norm.append(sn)
        else:
            flat_orig.append(orig_t)
            flat_norm.append(norm_str)
            
    # 3. Load Models
    lid_crf = joblib.load(LID_MODEL)
    pos_crf = joblib.load(POS_MODEL)

    # 4. Predict LID
    lid_features = lid_sent2features(flat_norm)
    lid_preds = lid_crf.predict([lid_features])[0]
    
    # 5. Predict POS
    pos_features = sent2pos_features(flat_norm, lid_preds)
    pos_preds = pos_crf.predict([pos_features])[0]

    print("\n--- Final Output ---")
    print(f"{'Token':<20} | {'Normalized':<20} | {'LID':<10} | {'POS':<10}")
    print("-" * 68)
    for orig, norm, lid, pos in zip(flat_orig, flat_norm, lid_preds, pos_preds):
        print(f"{orig:<20} | {norm:<20} | {lid:<10} | {pos:<10}")

if __name__ == "__main__":
    sample_text = "Bhai aaj ka mausam bahut awesome hai! Let's go for a walk."
    if len(sys.argv) > 1:
        sample_text = " ".join(sys.argv[1:])
    test_pipeline(sample_text)
