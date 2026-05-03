"""
matrix_classifier.py
Matrix Language Classifier for Hinglish code-mixed text.

Based on the Matrix Language Frame (MLF) model (Myers-Scotton, 1993).
Determines which language (HI or EN) provides the grammatical backbone
(Matrix Language) and which merely contributes vocabulary (Embedded Language).

Algorithm:
    1. Identify "anchor" tokens — grammatical function words detected via
       their POS tags (verbs, auxiliaries, conjunctions, determiners,
       pronouns, postpositions/prepositions).
    2. Each anchor contributes a weighted vote (default weight 2.0) toward
       its language's score; non-anchor content words contribute weight 1.0.
    3. If the two language scores are within 15 % of each other → BALANCED.
       Otherwise the higher-scoring language is the Matrix Language.

Usage:
    from hinglish.statistical_models.matrix_classifier import MatrixClassifier
    clf = MatrixClassifier()
    result = clf.classify(tokens, lid_labels, pos_tags)
    print(result["matrix_language"])   # "HI" | "EN" | "BALANCED"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── POS tag sets that mark grammatical function words ─────────────────────────

# HMM POS tags used in the Hinglish corpus that signal Hindi structure
HI_FUNCTION_POS: frozenset[str] = frozenset({
    "V",    # main verb
    "G_V",  # verb (Gyan-tagged)
    "PRT",  # particle / auxiliary
    "G_PRT",
    "PSP",  # postposition (Hindi-specific; very strong signal)
    "PRP",  # pronoun
    "G_PRP",
    "CC",   # coordinating conjunction
    "DT",   # determiner
    "G_N",  # gerund / verbal noun
})

# Penn-style / universal tags that signal English structure
EN_FUNCTION_POS: frozenset[str] = frozenset({
    "V",    # verb (shared, but context decides)
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",   # Penn verb forms
    "MD",   # modal auxiliary
    "IN",   # preposition / subordinating conjunction
    "DT",   # determiner
    "CC",   # coordinating conjunction
    "PRP",  # pronoun
    "WP",   # wh-pronoun
    "WDT",  # wh-determiner
    "WRB",  # wh-adverb
    "RP",   # particle
    "TO",   # infinitive marker
})

STRUCTURAL_WEIGHT  = 2.0   # weight for function-word anchors
CONTENT_WEIGHT     = 1.0   # weight for all other tokens
BALANCE_THRESHOLD  = 0.15  # if |hi - en| / max(hi, en) ≤ this → BALANCED


class MatrixClassifier:
    """Heuristic Matrix Language Classifier based on the MLF model."""

    def classify(
        self,
        tokens:     list[str],
        lid_labels: list[str],
        pos_tags:   list[str],
    ) -> dict:
        """
        Classify the matrix language of a code-mixed sentence.

        Parameters
        ----------
        tokens     : list of surface-form tokens
        lid_labels : LID label per token  ("HI" | "EN" | "UNI" | "EMOJI")
        pos_tags   : POS tag per token

        Returns
        -------
        dict with keys:
            matrix_language   : "HI" | "EN" | "BALANCED"
            embedded_language : "EN" | "HI" | "NONE"
            hi_score          : float
            en_score          : float
            evidence          : list[dict] — anchor tokens used for the decision
        """
        hi_score: float = 0.0
        en_score: float = 0.0
        evidence: list[dict] = []

        for tok, lid, pos in zip(tokens, lid_labels, pos_tags):
            if lid == "UNI" or lid == "EMOJI":
                continue  # neutral tokens do not vote

            is_hi_anchor = pos in HI_FUNCTION_POS and lid == "HI"
            is_en_anchor = pos in EN_FUNCTION_POS and lid == "EN"

            weight = STRUCTURAL_WEIGHT if (is_hi_anchor or is_en_anchor) else CONTENT_WEIGHT

            if lid == "HI":
                hi_score += weight
                if is_hi_anchor:
                    evidence.append({"token": tok, "lid": lid, "pos": pos, "weight": weight})
            elif lid == "EN":
                en_score += weight
                if is_en_anchor:
                    evidence.append({"token": tok, "lid": lid, "pos": pos, "weight": weight})

        # ── Decision ──────────────────────────────────────────────────────────
        total = hi_score + en_score
        if total == 0:
            return {
                "matrix_language":   "BALANCED",
                "embedded_language": "NONE",
                "hi_score":          0.0,
                "en_score":          0.0,
                "evidence":          [],
            }

        hi_ratio = hi_score / total
        en_ratio = en_score / total
        diff = abs(hi_ratio - en_ratio)

        if diff <= BALANCE_THRESHOLD:
            matrix   = "BALANCED"
            embedded = "NONE"
        elif hi_score > en_score:
            matrix   = "HI"
            embedded = "EN" if en_score > 0 else "NONE"
        else:
            matrix   = "EN"
            embedded = "HI" if hi_score > 0 else "NONE"

        return {
            "matrix_language":   matrix,
            "embedded_language": embedded,
            "hi_score":          round(hi_score, 3),
            "en_score":          round(en_score, 3),
            "evidence":          evidence,
        }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from hinglish.statistical_models.lid_model  import LIDModel
    from hinglish.statistical_models.hmm_pos_tagger import HMMPosTagger

    clf     = MatrixClassifier()
    lid_mdl = LIDModel();       lid_mdl.load()
    tagger  = HMMPosTagger();   tagger.load()

    sentences = [
        "Main school ja raha hoon",
        "I am going to the market today",
        "Yaar I cannot believe this hogaya",
        "kya hua bro sab theek hai",
    ]

    for sent in sentences:
        import re
        tokens = re.findall(r"[\w']+|[.,!?;]", sent.lower())
        lids   = lid_mdl.predict(tokens)
        pos    = [tag for _, tag in tagger.viterbi_decode(tokens)]
        result = clf.classify(tokens, lids, pos)

        print(f"\n{sent}")
        print(f"  Matrix   : {result['matrix_language']}")
        print(f"  Embedded : {result['embedded_language']}")
        print(f"  HI score : {result['hi_score']}   EN score : {result['en_score']}")
        if result["evidence"]:
            print(f"  Anchors  : {[e['token'] for e in result['evidence']]}")
