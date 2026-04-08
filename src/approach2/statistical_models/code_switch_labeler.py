import sys
import os
import string

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from approach2.features.phonetic_matcher import get_phonetic_hash

class CodeswitchSegmenter:
    def __init__(self):
        # Load indicators from generated wordlists
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        hi_path = os.path.join(data_dir, "hi_indicators.json")
        en_path = os.path.join(data_dir, "en_indicators.json")
        
        try:
            import json
            with open(hi_path, "r", encoding="utf-8") as f:
                self.hi_indicators = set(json.load(f))
            with open(en_path, "r", encoding="utf-8") as f:
                self.en_indicators = set(json.load(f))
        except FileNotFoundError:
            # Fallback to empty sets if not generated yet
            self.hi_indicators = set()
            self.en_indicators = set()
    
    def is_universal(self, token:str) -> bool:
        if not token.isalpha() or all(c in string.punctuation for c in token):
            return True
        return False
    
    def segment(self, tokens: list[str]) -> list[str]:
        raw_labels=[]

        from approach2.features.phonetic_matcher import fallback_dict

        for token in tokens:
            if self.is_universal(token):
                raw_labels.append("UNI")
                continue
                
            # If the word is in the fallback dictionary, map it to standard spelling
            tok_low = token.lower()
            norm_token = fallback_dict.get(tok_low, tok_low)

            if norm_token in self.hi_indicators:
                raw_labels.append("HI")
            elif norm_token in self.en_indicators:
                raw_labels.append("EN")    
            else:
                raw_labels.append("UNKNOWN")        
        
        final_labels = []
        for i, label in enumerate(raw_labels):
            if label != "UNKNOWN":
                final_labels.append(label)
            else:
                # Add boundary checks to prevent IndexError
                if i > 0 and i < len(raw_labels) - 1 and raw_labels[i-1] == "HI" and raw_labels[i+1] == "HI":
                    final_labels.append("HI")
                elif i > 0 and i < len(raw_labels) - 1 and raw_labels[i-1] == "EN" and raw_labels[i+1] == "EN":
                    final_labels.append("EN")
                else:
                    # Simply fallback to the previous valid label or HI
                    prev = final_labels[-1] if len(final_labels) > 0 else "HI"
                    final_labels.append(prev)
        return final_labels

if __name__ == "__main__":
    import re
    segmenter = CodeswitchSegmenter()

    # Use regex to properly separate punctuation from words
    raw_text = input("Enter Sentence: ")
    test_sentence = re.findall(r"[\w']+|[.,!?;]", raw_text)
    
    print("tokens: ", test_sentence)
    print("segments: ", segmenter.segment(test_sentence))