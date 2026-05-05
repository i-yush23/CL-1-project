"""
pipeline_runner.py — End-to-end Hinglish NLP Pipeline

Pipeline stages:
  1. Tokenise  (words + punctuation; emojis kept as single tokens)
  2. Emoji pre-processing  →  placeholder tokens
  3. Language Identification (LID)
  4. POS Tagging  (HMM Viterbi)
  5. Code-Mixing Index (CMI)
  6. Matrix Language Classification (MLF-based)

Run interactively:
    python hinglish/pipeline_runner.py

Run with a sentence argument:
    python hinglish/pipeline_runner.py "yaar I cannot believe this hogaya"
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hinglish.statistical_models.lid_model        import LIDModel
from hinglish.statistical_models.hmm_pos_tagger   import HMMPosTagger
from hinglish.statistical_models.matrix_classifier import MatrixClassifier
from hinglish.features.emoji_analyzer             import (
    is_emoji_token, emoji_placeholder, extract_emojis, EMOJI_RE
)
from hinglish.metrics.cmi_calculator              import CmiCalculator


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split on whitespace, keeping emojis and punctuation as own tokens."""
    tokens: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            tokens.append("".join(current))
            current.clear()

    keep_inside = {"_", "'", "-", "@", "#"}
    for ch in text:
        if ch.isspace():
            flush()
        elif is_emoji_token(ch):
            flush()
            tokens.append(ch)
        elif ch.isalnum() or ch in keep_inside:
            current.append(ch)
        else:
            flush()
            tokens.append(ch)
    flush()
    return [t for t in tokens if t]


def _preprocess_emojis(tokens: list[str]) -> tuple[list[str], list[str]]:
    """
    Replace emoji tokens with sentiment placeholders.
    Returns (processed_tokens, original_tokens).
    """
    processed = []
    for tok in tokens:
        if is_emoji_token(tok):
            processed.append(emoji_placeholder(tok))
        else:
            processed.append(tok)
    return processed, tokens


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(text: str) -> None:
    # ── load models ──────────────────────────────────────────────────────────
    lid_model  = LIDModel();         lid_model.load()
    tagger     = HMMPosTagger();     tagger.load()
    matrix_clf = MatrixClassifier()
    cmi_calc   = CmiCalculator()

    # ── stage 1 & 2 : tokenise + emoji pre-process ───────────────────────────
    raw_tokens              = _tokenize(text)
    proc_tokens, orig_tokens = _preprocess_emojis(raw_tokens)

    if not proc_tokens:
        print("[WARN] No tokens found in input.")
        return

    # ── stage 3 : LID ────────────────────────────────────────────────────────
    lid_labels = lid_model.predict(proc_tokens)

    # Override LID for emoji placeholders
    lid_labels = [
        "EMOJI" if tok.startswith("__EMOJI_") else lbl
        for tok, lbl in zip(proc_tokens, lid_labels)
    ]

    # ── stage 4 : POS ────────────────────────────────────────────────────────
    pos_results = tagger.viterbi_decode(proc_tokens)
    pos_tags    = [tag for _, tag in pos_results]

    # Override POS for emoji placeholders
    pos_tags = [
        "EMOJI" if tok.startswith("__EMOJI_") else tag
        for tok, tag in zip(proc_tokens, pos_tags)
    ]

    # ── stage 5 : CMI ────────────────────────────────────────────────────────
    cmi_result = cmi_calc.calculate(text)

    # ── stage 6 : Matrix Language ────────────────────────────────────────────
    matrix_result = matrix_clf.classify(proc_tokens, lid_labels, pos_tags)

    # ── display ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  INPUT : {text}")
    print(f"{'─'*55}")

    print(f"\n  {'Token':<22} {'LID':<7} {'POS'}")
    print(f"  {'─'*45}")
    for orig, proc, lid, pos in zip(orig_tokens, proc_tokens, lid_labels, pos_tags):
        display = orig if not orig.startswith("__EMOJI_") else proc
        print(f"  {display:<22} {lid:<7} {pos}")

    print(f"\n  {'─'*45}")
    print(f"  CMI Score        : {cmi_result['cmi']:.1f} / 100")
    print(f"  HI tokens        : {cmi_result['hi_count']}   "
          f"EN tokens: {cmi_result['en_count']}   "
          f"UNI tokens: {cmi_result['universal_count']}")

    print(f"\n  Matrix Language  : {matrix_result['matrix_language']}")
    print(f"  Embedded Language: {matrix_result['embedded_language']}")
    print(f"  HI score: {matrix_result['hi_score']}   EN score: {matrix_result['en_score']}")

    emojis_found = extract_emojis(text)
    if emojis_found:
        print(f"\n  Emojis detected  : {' '.join(emojis_found)}")

    if matrix_result["evidence"]:
        anchor_words = [e["token"] for e in matrix_result["evidence"]]
        print(f"  Grammar anchors  : {', '.join(anchor_words)}")

    print(f"{'─'*55}\n")


# ── entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Sentence passed as CLI argument
        run_pipeline(" ".join(sys.argv[1:]))
    else:
        # Interactive REPL
        print("╔══════════════════════════════════════════════════╗")
        print("║     Hinglish NLP Pipeline  (type 'quit' to exit) ║")
        print("╚══════════════════════════════════════════════════╝")
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                print("Bye!")
                break
            run_pipeline(text)
