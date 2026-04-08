"""
hindi_normalizer.py
Expands and normalizes common Romanized Hindi abbreviations and informal
contractions found in Hinglish social-media text.

These rules fire ONLY on tokens already classified as 'HI' by 
CodeMixedNormalizer, so there is zero risk of corrupting English words.
"""

import re

# ---------------------------------------------------------------------------
# Hindi Romanized expansion dictionary
# Common contractions, clippings, and phonetic shorthands used online
# ---------------------------------------------------------------------------
HINDI_EXPAND: dict[str, str] = {
    # Pronouns & common words
    "h":       "hai",
    "hai":     "hai",
    "hh":      "haha",
    "k":       "ka",
    "ki":      "ki",
    "ka":      "ka",
    "ko":      "ko",
    "m":       "mein",
    "me":      "mein",
    "mein":    "mein",
    "n":       "ne",
    "ne":      "ne",
    "se":      "se",
    "pr":      "par",
    "par":     "par",

    # Common verb clippings
    "rha":     "raha",
    "rhi":     "rahi",
    "rhe":     "rahe",
    "krna":    "karna",
    "kro":     "karo",
    "kru":     "karun",
    "lgta":    "lagta",
    "lgti":    "lagti",
    "dekh":    "dekho",
    "smjh":    "samjho",
    "bol":     "bol",
    "bta":     "batao",
    "btao":    "batao",
    "suno":    "suno",
    "aaja":    "aaja",
    "aao":     "aao",
    "jao":     "jao",

    # People / address
    "bhai":    "bhai",
    "bhaiya":  "bhaiya",
    "yaar":    "yaar",
    "yar":     "yaar",   # post-compress artifact fix
    "yrr":     "yaar",
    "yrrr":    "yaar",
    "dost":    "dost",
    "sir":     "sir",

    # Negative
    "nhi":     "nahi",
    "nai":     "nahi",
    "nah":     "nahi",
    "nahi":    "nahi",
    "nahin":   "nahi",
    "nahn":    "nahi",   # post-compress artifact fix
    "nahn":    "nahi",
    "na":      "nahi",
    "mat":     "mat",

    # Question words
    "kya":     "kya",
    "kyu":     "kyun",
    "kyun":    "kyun",
    "kaise":   "kaise",
    "kahan":   "kahan",
    "kb":      "kab",
    "kab":     "kab",
    "kaun":    "kaun",

    # Common adjectives / adverbs
    "sahi":    "sahi",
    "glt":     "galat",
    "galat":   "galat",
    "accha":   "achha",
    "acha":    "achha",
    "achha":   "achha",
    "thk":     "theek",
    "thik":    "theek",
    "theek":   "theek",
    "bhut":    "bahut",
    "bht":     "bahut",
    "bahut":   "bahut",
    "zyada":   "zyada",
    "thoda":   "thoda",

    # Common expressions
    "hn":      "haan",
    "haa":     "haan",
    "haan":    "haan",
    "han":     "haan",
    "hm":      "haan",
    "ji":      "ji",
    "nope":    "nope",
    "yrr":     "yaar",
    "yrrr":    "yaar",
    "bc":      "bahenchod",
    "mc":      "madarchod",
    "abey":    "abey",
    "abe":     "abey",
    "ary":     "are",
    "are":     "are",
    "arrey":   "are",
    "waise":   "waise",
    "vaise":   "waise",
    "matlab":  "matlab",
    "mtlb":    "matlab",
    "smjha":   "samjha",
    "pta":     "pata",
    "pata":    "pata",
    "lga":     "laga",
    "lagi":    "lagi",
    "lag":     "laga",
}

# ---------------------------------------------------------------------------
# Elongation pattern (re-used from normalizer)
# ---------------------------------------------------------------------------
_ELONGATION_RE = re.compile(r'(.)\1{2,}')


def compress_hindi(token: str) -> str:
    """
    Compress elongated Romanized Hindi tokens.
    Strategy: reduce 3+ repeats to 1 (not 2), because Hindi root forms
    usually don't have double letters.
    e.g.  'bhaaaaai' → 'bhai'
          'yaaaar'   → 'yaar' (kept as 2 since 'yaar' has double-a)
    """
    # Reduce 3+ repeats down to 1 first
    compressed = re.sub(r'(.)\1{2,}', r'\1', token)
    return compressed


def expand_hindi(token: str) -> str:
    """
    Look up a token (after compression) in the Hindi expansion dictionary.
    Returns the canonical form, or the compressed token if not found.
    """
    tok = token.lower()
    compressed = compress_hindi(tok)

    # Try the compressed form first, then the original lowercased
    return HINDI_EXPAND.get(compressed, HINDI_EXPAND.get(tok, compressed))


def normalize_hindi_token(token: str) -> str:
    """
    Full normalization for a single Romanized Hindi token:
      compress elongation → expand abbreviation/contraction.
    """
    if not token.isalpha():
        return token
    return expand_hindi(token)


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tokens = [
        # Elongation cases
        "bhaaaaai", "yaaaar", "bahutttt", "nahhiii",
        # Abbreviation cases
        "rha", "rhi", "nhi", "kyu", "lgta", "krna", "bht", "thk",
        # Already canonical
        "bhai", "yaar", "kya", "sahi",
        # Edge cases
        "h", "m", "n", "k",
    ]

    print(f"{'Input':<15} → {'Output'}")
    print("-" * 30)
    for tok in test_tokens:
        print(f"{tok:<15} → {normalize_hindi_token(tok)}")
