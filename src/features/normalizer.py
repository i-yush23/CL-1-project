"""
normalizer.py
HinglishNormalizer: rule-based text normalization for Hinglish social-media text.

Steps applied (in order):
  1. Lowercase.
  2. Expand common Hinglish abbreviations / SMS-slang.
  3. Compress elongated characters (e.g. 'helloooo' → 'hello').
  4. Basic English spell-check on purely alphabetic tokens.
"""

import re
from spellchecker import SpellChecker


# ---------------------------------------------------------------------------
# Abbreviation / slang dictionary
# ---------------------------------------------------------------------------
SLANG_DICT: dict[str, str] = {
    # General internet / SMS slang
    "gm":    "good morning",
    "gn":    "good night",
    "lol":   "laughing out loud",
    "omg":   "oh my god",
    "brb":   "be right back",
    "tbh":   "to be honest",
    "imo":   "in my opinion",
    "ngl":   "not gonna lie",
    "idk":   "i don't know",
    "gg":    "good game",
    # Hinglish-specific
    "bc":    "bahenchod",      # common Hinglish expletive abbreviation (flagged)
    "bhai":  "bhai",           # already normalised form – keep as-is anchor
    "kya":   "kya",
    "nahi":  "nahi",
    "h":     "hai",            # very common 'h' → 'hai' in Hinglish
    "k":     "okay",
    "hbu":   "how about you",
    "wbu":   "what about you",
    "ty":    "thank you",
    "np":    "no problem",
}

# Regex: match whole word only, case-insensitive
_SLANG_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in SLANG_DICT) + r')\b',
    re.IGNORECASE,
)

# Elongation: 3+ repeated characters → at most 2
_ELONGATION_RE = re.compile(r'(.)\1{2,}')


class HinglishNormalizer:
    """
    Rule-based normalizer for Hinglish tokens.

    Parameters
    ----------
    spell_check : bool
        Whether to apply English spell-checking on alphabetic tokens.
        Disable when running over a dataset that contains Hindi Romanized
        words to avoid false corrections.
    """

    def __init__(self, spell_check: bool = True):
        self._spell = SpellChecker() if spell_check else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_token(self, token: str) -> str:
        """Normalize a *single* token and return the result."""
        # Skip protected spans (hashtags, emojis, URLs, @mentions)
        if token.startswith("#") or token.startswith("@") or token.startswith("http"):
            return token
        # Skip non-alphabetic tokens (punctuation, numbers, emojis)
        if not token.isalpha():
            return token

        tok = token.lower()
        tok = self._expand_slang(tok)
        tok = self._compress_elongation(tok)
        tok = self._spell_correct(tok)
        return tok

    def normalize_sentence(self, tokens: list[str]) -> list[str]:
        """Normalize a *list* of tokens (output of HinglishTokenizer)."""
        return [self.normalize_token(t) for t in tokens]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _expand_slang(self, tok: str) -> str:
        """Replace known abbreviations / slang with their expanded form."""
        return _SLANG_RE.sub(lambda m: SLANG_DICT[m.group(0).lower()], tok)

    def _compress_elongation(self, tok: str) -> str:
        """
        Compress runs of 3+ identical characters down to 2.
        'helloooo' → 'helloo'  (a second pass of real spell-check
        will reduce 'helloo' → 'hello').
        """
        return _ELONGATION_RE.sub(r'\1\1', tok)

    def _spell_correct(self, tok: str) -> str:
        """
        Attempt an English spell-correction.
        Returns the most probable correction, or the original word
        if spell checker has no suggestion or token is not alphabetic.
        """
        if self._spell is None or not tok.isalpha():
            return tok
        correction = self._spell.correction(tok)
        return correction if correction else tok


# ── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    norm = HinglishNormalizer(spell_check=True)
    samples = [
        ["helloooo", "bhai", "kya", "haal", "h", "??"],
        ["omg", "sooooo", "cute", "lol"],
        ["@RahulG", "#India", "wbu", "ngl", "idk"],
        ["bhaaaaai", "tu", "sahi", "bol", "rha"],
    ]
    for toks in samples:
        print(f"\nInput : {toks}")
        print(f"Output: {norm.normalize_sentence(toks)}")
