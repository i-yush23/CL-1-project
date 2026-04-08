import os
import sys
import re

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from spellchecker import SpellChecker
from src.approach1.features.hindi_normalizer import normalize_hindi_token

# ---------------------------------------------------------------------------
# Configuration & Dictionaries
# ---------------------------------------------------------------------------
HINGLISH_SLANG: dict[str, str] = {
    "bc":    "bahenchod",
    "h":     "hai",
    "k":     "okay",
    "bhai":  "bhai",
    "kya":   "kya",
    "nahi":  "nahi",
}

ENGLISH_SLANG: dict[str, str] = {
    "gm":    "good morning",
    "gn":    "good night",
    "lol":   "laughing out loud",
    "omg":   "oh my god",
    "brb":   "be right back",
    "tbh":   "to be honest",
    "imo":   "in my opinion",
    "ngl":   "not gonna lie",
    "idk":   "i don't know",
    "wbu":   "what about you",
    "hbu":   "how about you",
    "ty":    "thank you",
    "np":    "no problem",
}

_ELONGATION_RE = re.compile(r'(.)\1{2,}')

class CodeMixedNormalizer:
    """
    A two-stage normalizer that tags language first, then applies 
    language-specific normalization rules.
    """

    def __init__(self):
        # Initialize English spellchecker
        self._spell = SpellChecker()
        # We combine slang for quick lookup, but can separate them in logic
        self.all_slang = {**HINGLISH_SLANG, **ENGLISH_SLANG}

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_label(self, token: str) -> str:
        """Identify if a token is English (EN), Hinglish (HI), or OTHER."""
        if not token.isalpha():
            return "OTHER"
        
        tok_low = token.lower()

        # 1. Check if it's a known English slang
        if tok_low in ENGLISH_SLANG:
            return "EN"
            
        # 2. Check if it's a known Hinglish slang
        if tok_low in HINGLISH_SLANG:
            return "HI"

        # 3. Check English Dictionary (after basic compression)
        # We compress 'helloooo' to 'helloo' just to see if the root is English
        test_tok = _ELONGATION_RE.sub(r'\1\1', tok_low)
        if self._spell.known([test_tok]):
            return "EN"

        # 4. Default to Hinglish for alphabetic unknowns
        return "HI"

    def _compress(self, tok: str) -> str:
        """Generic compression: 3+ chars to 2."""
        return _ELONGATION_RE.sub(r'\1\1', tok)

    # ------------------------------------------------------------------
    # Normalization Engines
    # ------------------------------------------------------------------

    def normalize_english(self, token: str) -> str:
        """Rules: Slang Expand -> Compress -> Spell Check."""
        tok = token.lower()
        if tok in ENGLISH_SLANG:
            return ENGLISH_SLANG[tok]
        
        tok = self._compress(tok)
        # Only spell-check if it's not already a perfect dictionary match
        if not self._spell.known([tok]):
            correction = self._spell.correction(tok)
            return correction if correction else tok
        return tok

    def normalize_hinglish(self, token: str) -> str:
        """Delegates to hindi_normalizer: compress elongation → expand abbrev."""
        return normalize_hindi_token(token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, tokens: list[str]) -> list[dict]:
        """
        Returns a list of dictionaries containing metadata for each token.
        """
        output = []
        for t in tokens:
            # Skip protected spans
            if t.startswith(("#", "@", "http")):
                output.append({"orig": t, "label": "SOCIAL", "norm": t})
                continue

            label = self._get_label(t)
            
            if label == "EN":
                norm = self.normalize_english(t)
            elif label == "HI":
                norm = self.normalize_hinglish(t)
            else:
                norm = t # Punctuation/Numbers
                
            output.append({
                "orig": t,
                "label": label,
                "norm": norm
            })
        return output

# ── Demo Execution ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    normalizer = CodeMixedNormalizer()
    
    test_sentences = [
        ["helloooo", "bhai", "kya", "haal", "h", "?"],
        ["omg", "this", "is", "sooooo", "sahi", "yaar"],
        ["bhaaaaai", "tu", "bhaji", "khayega", "?"],
        ["helloooo", "bhai", "kya", "haal", "h", "??"],
        ["omg", "sooooo", "cute", "lol"],
        ["@RahulG", "#India", "wbu", "ngl", "idk"],
        ["bhaaaaai", "tu", "sahi", "bol", "rha"]
    ]

    for sentence in test_sentences:
        print(f"\nProcessing: {' '.join(sentence)}")
        results = normalizer.process(sentence)
        
        # Display results in a readable format
        for item in results:
            print(f"  {item['orig']:12} | Label: {item['label']:6} | Norm: {item['norm']}")