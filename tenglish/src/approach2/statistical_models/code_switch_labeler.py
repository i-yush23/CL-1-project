 

import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from approach2.features.emoji_analyzer import is_emoji_token
from approach2.features.phonetic_matcher import get_phonetic_hash, fallback_dict

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Common English contractions — classify straight to EN
_EN_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "shouldn't",
    "wouldn't", "isn't", "aren't", "wasn't", "weren't", "it's", "i'm", "i've",
    "i'd", "i'll", "you're", "you've", "you'd", "you'll", "he's", "she's",
    "we're", "we've", "we'd", "we'll", "they're", "they've", "they'd",
    "that's", "what's", "there's", "here's", "let's",
}


def _normalize_elongated(word: str) -> str:
    """Collapse 3+ repeated chars to 2 (yaaarr → yaar, pleaaase → please)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', word)


class CodeswitchSegmenter:

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = _DATA_DIR

        te_path = os.path.join(data_dir, "te_indicators.json")
        en_path = os.path.join(data_dir, "en_indicators.json")

        try:
            with open(te_path, "r", encoding="utf-8") as f:
                self.te_indicators: set[str] = set(json.load(f))
        except FileNotFoundError:
            print(f"[WARN] te_indicators.json not found. Run tools/regenerate_indicators.py first.")
            self.te_indicators = set()

        try:
            with open(en_path, "r", encoding="utf-8") as f:
                self.en_indicators: set[str] = set(json.load(f))
        except FileNotFoundError:
            print(f"[WARN] en_indicators.json not found. Run tools/regenerate_indicators.py first.")
            self.en_indicators = set()

        self._te_hashes: set[str] = {get_phonetic_hash(w) for w in self.te_indicators}
        self._en_hashes: set[str] = {get_phonetic_hash(w) for w in self.en_indicators}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_universal(self, token: str) -> bool:
        """True for punctuation, pure-digit tokens, URLs, hashtags, mentions."""
        if not token:
            return True
        if token.startswith(("#", "@", "http")):
            return True
        if token.isdigit():
            return True
        # purely non-alpha (punctuation / symbols) but NOT alphanumeric mixes
        if all(c in string.punctuation or not c.isalpha() for c in token):
            return True
        return False

    def _classify_one(self, token: str) -> str:
        """Classify a single alpha token; returns 'TE', 'EN', or 'UNKNOWN'."""
        # English contractions
        if token.lower() in _EN_CONTRACTIONS:
            return "EN"

        tok_low = token.lower()

        # Normalize elongated tokens (yaaarr → yaar)
        tok_norm = _normalize_elongated(tok_low)

        # Transliteration normalization
        norm = fallback_dict.get(tok_norm, tok_norm)

        # Direct indicator lookup
        if norm in self.te_indicators:
            return "TE"
        if norm in self.en_indicators:
            return "EN"

        # Phonetic hash fallback
        ph = get_phonetic_hash(norm)
        if ph in self._te_hashes:
            return "TE"
        if ph in self._en_hashes:
            return "EN"

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, tokens: list[str]) -> list[str]:
        """
        Returns a list of LID labels parallel to *tokens*.
        Labels: 'TE', 'EN', 'UNI', 'EMOJI'
        """
        raw_labels: list[str] = []
        for token in tokens:
            if is_emoji_token(token):
                raw_labels.append("EMOJI")
            elif self._is_universal(token):
                raw_labels.append("UNI")
            else:
                raw_labels.append(self._classify_one(token))

        # Context smoothing: resolve UNKNOWN using wider window (±2 neighbours)
        final_labels: list[str] = []
        for i, label in enumerate(raw_labels):
            if label != "UNKNOWN":
                final_labels.append(label)
                continue

            # Collect up to 2 left and 2 right non-UNI/non-UNKNOWN neighbours
            candidates: list[str] = []
            for offset in range(1, 3):
                if i - offset >= 0 and raw_labels[i - offset] not in ("UNKNOWN", "UNI", "EMOJI"):
                    candidates.append(raw_labels[i - offset])
                if i + offset < len(raw_labels) and raw_labels[i + offset] not in ("UNKNOWN", "UNI", "EMOJI"):
                    candidates.append(raw_labels[i + offset])

            if candidates:
                # Majority vote among neighbours
                te_count = candidates.count("TE")
                en_count = candidates.count("EN")
                if te_count >= en_count:
                    final_labels.append("TE")
                else:
                    final_labels.append("EN")
            else:
                # Default: assume Telugu
                final_labels.append("TE")

        return final_labels


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    segmenter = CodeswitchSegmenter()
    raw = input("Enter a Tenglish sentence: ")
    tokens = re.findall(r"[\w']+|[.,!?;]", raw)

    print(f"\n{'Token':<20} {'LID'}")
    print("-" * 28)
    labels = segmenter.segment(tokens)
    for tok, lbl in zip(tokens, labels):
        print(f"{tok:<20} {lbl}")
