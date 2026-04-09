"""
phonetic_matcher.py  (improved)
Phonetic hashing for Tenglish (Telugu-English) tokens.

Key fixes vs original:
  - BUG FIX: regex back-reference was r'\\1' (literal backslash) → now r'\1'
    so duplicate-char compression (aa→a, dd→d …) actually works.
  - Added common Telugu pre-nasalised clusters (nd, mb, nk, nj).
  - Added 'j' and 'y' to the phonetic map for consistency.
"""

import re
import json
import os
from collections import defaultdict


def get_phonetic_hash(word: str) -> str:
    """
    Converts a Tenglish word into its phonetic skeleton to match spelling variants.
    e.g., 'cheppadu', 'chepadu', 'chpdu' -> same hash
          'vaadu', 'vadu', 'vaaduu' -> same hash
    """
    if not word or not word.isalpha():
        return word.lower()

    w = word.lower()

    # 1. Compress repeated characters (aa->a, ee->e, oo->o, dd->d, tt->t …)
    #    BUG FIX: original had r'\\1' (literal backslash) breaking this entirely.
    w = re.sub(r'(.)\1+', r'\1', w)

    # 2. Multi-character clusters (order matters — longer patterns first)
    w = w.replace("tsh", "C")
    w = w.replace("ch",  "C")
    w = w.replace("tch", "C")
    w = w.replace("sh",  "S")
    w = w.replace("ph",  "F")
    w = w.replace("bh",  "B")
    w = w.replace("kh",  "K")
    w = w.replace("gh",  "G")
    w = w.replace("th",  "T")
    w = w.replace("dh",  "D")
    w = w.replace("jh",  "J")
    w = w.replace("ng",  "N")
    # Pre-nasalised clusters common in Telugu
    w = w.replace("nj",  "J")
    w = w.replace("mb",  "B")
    w = w.replace("nd",  "D")
    w = w.replace("nk",  "K")

    # 3. Single-character phonetic groupings
    phonetic_map = {
        'k': 'K', 'c': 'K', 'q': 'K',
        'v': 'W', 'w': 'W',
        's': 'S', 'z': 'S',
        'f': 'F',
        'x': 'K',
        'j': 'J',
        'y': 'Y',
    }
    result = []
    for ch in w:
        result.append(phonetic_map.get(ch, ch))
    w = "".join(result)

    # 4. Drop vowels from non-initial position
    if len(w) <= 1:
        return w.upper()

    first_char = w[0]
    remaining  = w[1:]
    remaining_no_vowels = re.sub(r'[aeiouy]', '', remaining)

    return (first_char + remaining_no_vowels).upper()


# ---------------------------------------------------------------------------
# Fallback transliteration dictionary
# ---------------------------------------------------------------------------
_fallback_path = os.path.join(os.path.dirname(__file__), "..", "data", "transliteration_variants.json")
try:
    with open(_fallback_path, "r", encoding="utf-8") as _f:
        fallback_dict: dict[str, str] = json.load(_f)
except FileNotFoundError:
    fallback_dict: dict[str, str] = {}


def normalize_word(word: str) -> str:
    word_lower = word.lower()
    if word_lower in fallback_dict:
        return fallback_dict[word_lower]
    return get_phonetic_hash(word_lower)


# ---------------------------------------------------------------------------
# CLI: regenerate phonetic bigram/trigram tables (unchanged)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from tqdm import tqdm
    current_dir = os.path.dirname(__file__)
    phonetic_bigrams: dict = defaultdict(lambda: defaultdict(float))
    bigram_file = os.path.join(current_dir, "bigram_probs.json")
    if os.path.exists(bigram_file):
        with open(bigram_file, "r") as f:
            bigram_probs = json.load(f)
        for prev_word, word_dict in tqdm(bigram_probs.items(), desc="Bigrams"):
            for next_word, prob in word_dict.items():
                phonetic_bigrams[normalize_word(prev_word)][normalize_word(next_word)] += prob
        for ph, nd in phonetic_bigrams.items():
            row_sum = sum(nd.values())
            if row_sum > 0:
                for nw in nd:
                    phonetic_bigrams[ph][nw] /= row_sum
        with open(os.path.join(current_dir, "phonetic_bigram_probs.json"), "w") as f:
            json.dump(phonetic_bigrams, f, indent=2)
        print("Saved phonetic_bigram_probs.json")
