import os
import sys
import joblib
import string

# ── Paths & Setup ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# Ensure these imports pull from the files we updated with Lexicon/Suffix features
import src.approach1.models.lid_crf as lid_crf
from src.approach1.models.lid_crf import sent2features as lid_sent2features
import src.approach1.models.pos_crf as pos_crf
from src.approach1.models.pos_crf import sent2pos_features

LID_MODEL_PATH = os.path.join(ROOT, "src", "approach1", "models", "lid_model.pkl")
POS_MODEL_PATH = os.path.join(ROOT, "src", "approach1", "models", "pos_model.pkl")

# ── Load Models Once (Global) ────────────────────────────────────────────────
try:
    LID_MODEL = joblib.load(LID_MODEL_PATH)
    POS_MODEL = joblib.load(POS_MODEL_PATH)
except FileNotFoundError:
    print("Error: Model files not found. Ensure you have trained lid_crf.py and pos_crf.py.")
    sys.exit(1)

def test_pipeline(text: str):
    from src.approach1.features.tokenizer import HinglishTokenizer
    from src.approach1.features.normalizer import CodeMixedNormalizer
    print(f"\n--- Running Optimized Hinglish Pipeline ---")
    print(f"Input: {text}")
    
    # 1. Tokenize
    tokenizer = HinglishTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # 2. Normalize & Smart Flattening
    # Optimization: We prevent expansions like 'tbh' -> 'to be honest' 
    # if they weren't in the training set, as they break CRF context windows.
    normalizer = CodeMixedNormalizer()
    norm_dicts = normalizer.process(tokens)
    
    flat_orig = []
    flat_norm = []
    for orig_t, d in zip(tokens, norm_dicts):
        norm_str = d["norm"]
        # If expansion results in more than 1 word, we check for 'slang' preservation
        if len(norm_str.split()) > 1:
            # We keep the original slang token (e.g., 'tbh') to match CRF training context
            flat_orig.append(orig_t)
            flat_norm.append(orig_t.lower()) 
        else:
            flat_orig.append(orig_t)
            flat_norm.append(norm_str)

    # 3. Predict Language ID (LID)
    # The feature extractor here now uses the Lexicon and 4-char suffix logic
    lid_features = lid_sent2features(flat_norm)
    lid_preds = LID_MODEL.predict([lid_features])[0]
    
    # 4. Predict POS Tags
    # The POS model now consumes the LID predictions as a guiding feature
    pos_features = sent2pos_features(flat_norm, lid_preds)
    pos_preds = POS_MODEL.predict([pos_features])[0]

    # 5. Accuracy Post-Processing (Heuristic Layer)
    # This catches common statistical slips for frequent Hinglish particles
    final_lid = []
    final_pos = []
    
    # Manual overrides for "unbreakable" rules
    overrides = {
        "se": ("HI", "PSP"),
        "ni": ("HI", "PRT"),
        "yr": ("HI", "PRT"),
        "baar": ("HI", "N"),
        "tbh": ("EN", "ADV")
    }

    for i, token in enumerate(flat_norm):
        t_low = token.lower()
        
        # Rule 1: Punctuation/Emoji check
        if all(c in string.punctuation for c in token):
            final_lid.append("univ")
            final_pos.append("PUNCT")
        
        # Rule 2: Social Media Mentions/Hashtags
        elif token.startswith(("@", "#")):
            final_lid.append("univ")
            final_pos.append("PROPN")
            
        # Rule 3: Lexicon Overrides
        elif t_low in overrides:
            final_lid.append(overrides[t_low][0])
            final_pos.append(overrides[t_low][1])
            
        # Rule 4: Statistical Model Prediction
        else:
            final_lid.append(lid_preds[i])
            final_pos.append(pos_preds[i])

    # 6. Output Formatting
    print("\n--- Final Pipeline Output ---")
    print(f"{'Original':<15} | {'Normalized':<15} | {'LID':<8} | {'POS':<8}")
    print("-" * 60)
    for orig, norm, lid, pos in zip(flat_orig, flat_norm, final_lid, final_pos):
        print(f"{orig:<15} | {norm:<15} | {lid:<8} | {pos:<8}")

if __name__ == "__main__":
    # Test cases that previously failed: 'yr', 'tbh', and 'se'
    sample_text = "tbh yr ye statistical models se kuch ni hona"
    if len(sys.argv) > 1:
        sample_text = " ".join(sys.argv[1:])
    test_pipeline(sample_text)