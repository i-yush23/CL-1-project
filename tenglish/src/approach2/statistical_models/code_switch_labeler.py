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

# Telugu Unicode block: U+0C00–U+0C7F
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# Tokens that are mostly digits/symbols with a few letters (100rs, 2pm, no1)
_MOSTLY_NUMERIC_RE = re.compile(r"^[0-9@#$%&*+\-./\\:;=?^_`|~]*[a-zA-Z]{0,3}[0-9]*$")


def _normalize_elongated(word: str) -> str:
    """Collapse 4+ repeated chars to 2 (yaaaaar → yaar), but leave 3-char runs
    untouched so genuine geminate Telugu romanisations (cheyyi, nuvvu) survive."""
    return re.sub(r'(.)\1{3,}', r'\1\1', word)


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
        """True for punctuation, pure-digit tokens, URLs, hashtags, mentions,
        and mixed alphanumeric tokens that carry no language signal."""
        if not token:
            return True
        if token.startswith(("#", "@", "http")):
            return True
        if token.isdigit():
            return True
        # purely non-alpha (punctuation / symbols)
        if all(c in string.punctuation or not c.isalpha() for c in token):
            return True
        # mixed tokens like 100rs, 2pm, no1, v2, r2d2
        if _MOSTLY_NUMERIC_RE.match(token.lower()):
            return True
        return False

    def _classify_one(self, token: str) -> str:
        """Classify a single alpha token; returns 'TE', 'EN', or 'UNKNOWN'."""

        # HARD RULE 1: any Telugu Unicode character → TE immediately
        if _TELUGU_RE.search(token):
            return "TE"

        # English contractions
        if token.lower() in _EN_CONTRACTIONS:
            return "EN"

        tok_low = token.lower()

        # Normalize elongated tokens (yaaaaar → yaar, but nuvvu stays nuvvu)
        tok_norm = _normalize_elongated(tok_low)

        # Transliteration normalization
        norm = fallback_dict.get(tok_norm, tok_norm)

        # Direct indicator lookup
        if norm in self.te_indicators:
            return "TE"
        if norm in self.en_indicators:
            return "EN"

        # Also try the un-normalised form (catches elongated forms like "heyyyy"
        # where the normalised "heyy" might not be in the EN list but "hey" is)
        norm_extra = fallback_dict.get(tok_low, tok_low)
        if norm_extra in self.te_indicators:
            return "TE"
        if norm_extra in self.en_indicators:
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

        # Context smoothing: resolve UNKNOWN using wider window (±3 neighbours)
        # Collect ALL neighbours first, then take a weighted majority vote.
        final_labels: list[str] = []
        for i, label in enumerate(raw_labels):
            if label != "UNKNOWN":
                final_labels.append(label)
                continue

            # Weighted vote: closer neighbours count more (weight = 3 - distance)
            te_score = 0
            en_score = 0
            for offset in range(1, 4):
                weight = 3 - offset  # offset 1→2, offset 2→1, offset 3→0
                for idx in (i - offset, i + offset):
                    if 0 <= idx < len(raw_labels):
                        nb = raw_labels[idx]
                        if nb == "TE":
                            te_score += weight
                        elif nb == "EN":
                            en_score += weight

            if te_score == 0 and en_score == 0:
                # No informative neighbours — look at sentence-level majority
                te_total = raw_labels.count("TE")
                en_total = raw_labels.count("EN")
                final_labels.append("TE" if te_total >= en_total else "EN")
            else:
                final_labels.append("TE" if te_score >= en_score else "EN")

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
