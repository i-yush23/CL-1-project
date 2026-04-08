
import json
import os
from collections import defaultdict

def regenerate_indicators(data_path="hinglish_crf_train_data.jsonl", output_dir="src/approach2/data"):
    word_label_counts = defaultdict(lambda: {"HI": 0, "EN": 0, "OTHER": 0})
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                tokens = data.get("tokens", [])
                labels = data.get("labels", [])
                for token, label in zip(tokens, labels):
                    low_token = token.lower()
                    if label in ["HI", "EN"]:
                        word_label_counts[low_token][label] += 1
                    else:
                        word_label_counts[low_token]["OTHER"] += 1
            except Exception as e:
                print(f"Error parsing line: {e}")

    hi_indicators = []
    en_indicators = []
    
    # Threshold: Word must appear at least twice, and be primarily one language (> 60% and > current count of other)
    for word, counts in word_label_counts.items():
        total = sum(counts.values())
        if total < 2: continue
        
        hi_v = counts["HI"]
        en_v = counts["EN"]
        
        if hi_v > en_v and hi_v / total > 0.6:
            hi_indicators.append(word)
        elif en_v > hi_v and en_v / total > 0.6:
            en_indicators.append(word)

    # Add common missing words if needed
    # (Optional: although they should be in the training data anyway)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "hi_indicators.json"), "w", encoding="utf-8") as f:
        json.dump(hi_indicators, f, indent=2)
    with open(os.path.join(output_dir, "en_indicators.json"), "w", encoding="utf-8") as f:
        json.dump(en_indicators, f, indent=2)

    print(f"Regenerated indicators. HI Words: {len(hi_indicators)}, EN Words: {len(en_indicators)}")

if __name__ == "__main__":
    regenerate_indicators()
