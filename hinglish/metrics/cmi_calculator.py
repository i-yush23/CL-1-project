"""
cmi_calculator.py
Code-Mixing Index (CMI) Calculator for Hinglish text.

CMI Formula:
    CMI = 100 * (1 - max(w_HI, w_EN) / (N - U))

Where:
    N        = total number of tokens
    U        = number of Universal tokens (punctuation, numbers, etc.)
    w_HI     = count of Hindi (HI) tokens
    w_EN     = count of English (EN) tokens

A score of 0 means the text is fully monolingual (ignoring punctuation/numbers).
A score of 100 means perfectly balanced code-mixing between two languages.

Usage:
    python hinglish/metrics/cmi_calculator.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hinglish.statistical_models.lid_model import LIDModel


class CmiCalculator:
    """Computes the Code-Mixing Index for a piece of Hinglish text."""

    def __init__(self):
        self.lid = LIDModel()
        self.lid.load()

    def tokenize(self, text: str) -> list[str]:
        """Basic whitespace + punctuation tokenizer (lowercased)."""
        return re.findall(r"[\w']+|[.,!?;]", text.lower())

    def calculate(self, text: str) -> dict:
        """
        Calculate the CMI for the given text.

        Returns
        -------
        dict with keys:
            cmi             : float   — CMI score in [0, 100]
            total_tokens    : int     — total token count (including UNI)
            hi_count        : int     — number of HI tokens
            en_count        : int     — number of EN tokens
            universal_count : int     — number of UNI tokens
            token_labels    : list    — list of (token, label) pairs
        """
        tokens = self.tokenize(text)

        if not tokens:
            return {
                "cmi": 0.0,
                "total_tokens": 0,
                "hi_count": 0,
                "en_count": 0,
                "universal_count": 0,
                "token_labels": [],
            }

        labels = self.lid.predict(tokens)
        label_counts = Counter(labels)

        w_hi = label_counts.get("HI", 0)
        w_en = label_counts.get("EN", 0)
        u    = label_counts.get("UNI", 0)
        n    = len(tokens)

        denominator = n - u
        cmi = 0.0 if denominator == 0 else 100.0 * (1 - max(w_hi, w_en) / denominator)

        return {
            "cmi": round(cmi, 2),
            "total_tokens": n,
            "hi_count": w_hi,
            "en_count": w_en,
            "universal_count": u,
            "token_labels": list(zip(tokens, labels)),
        }


if __name__ == "__main__":
    calc = CmiCalculator()

    test_sentences = [
        "Main school ja raha hoon",
        "I am going to the market today",
        "Yaar I cannot believe this hogaya",
        "kya hua bro, sab theek hai?",
        "tbh yr ye statistical models se kuch ni hona",
    ]

    print(f"\n{'Sentence':<50} {'CMI':>6}  Breakdown")
    print("-" * 75)
    for sent in test_sentences:
        r = calc.calculate(sent)
        print(
            f"{sent:<50} {r['cmi']:>6.1f}  "
            f"HI={r['hi_count']} EN={r['en_count']} UNI={r['universal_count']}"
        )
