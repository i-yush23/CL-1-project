"""
demo_features.py  —  Approach 2: Purely Statistical Pipeline
=============================================================
Run this to see which Approach-2 components are active and what
each one contributes.  Toggle FEATURE_FLAGS below (or pass --level N)
to control which stages participate.

Feature Levels (add-on):
  0  — Raw tokenised input (baseline, always on)
  1  — Transliteration Variants Fallback  [Component C]
  2  — Phonetic Matcher                   [Component B]
  3  — N-gram / Bigram Analyser           [Component A]
  4  — Code-Switch Segmenter              [Component D]
  5  — Emoji Analyser                     [Component E]

Each level is skipped gracefully if its module has not been
implemented yet, so you can run this at any point in development.
"""

import os
import sys
import importlib
import argparse

# ── Make project root importable ──────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Feature registry ──────────────────────────────────────────────────────────
# Each entry:  level → (label, module_path, callable_name, description)
#   module_path  : dotted import path relative to project root
#   callable_name: function/class to call;  None → check module import only
FEATURE_REGISTRY = {
    1: (
        "Transliteration Variants Fallback",
        "src.approach2.features.transliteration_variants",
        "apply_variants",
        "Static dict mapping common spelling variants to canonical forms\n"
        "     (e.g. nhi→nahi, bht→bahut, yr→yaar). Always checked first.",
    ),
    2: (
        "Phonetic Matcher",
        "src.approach2.features.phonetic_matcher",
        "apply_phonetic",
        "Custom Hinglish Metaphone: drops medial vowels, compresses repeated\n"
        "     consonants, groups equivalent sounds (k/c/q→K, v/w→W), then\n"
        "     resolves fuzzy matches via Levenshtein with fine-tuned weights.",
    ),
    3: (
        "N-gram / Bigram Analyser",
        "src.approach2.features.ngram_analyzer",
        "apply_ngram",
        "Viterbi-style maximum-likelihood correction using pre-built bigram\n"
        "     and trigram probability tables from the training corpus.",
    ),
    4: (
        "Code-Switch Segmenter",
        "src.approach2.statistical_models.code_switch_labeler",
        "apply_segmenter",
        "Rolls a 3-word window to score HI vs EN 'mass', then emits\n"
        "     segment spans ([0:4]→HI, [4:7]→EN) with transition penalties.",
    ),
    5: (
        "Emoji Analyser",
        "src.approach2.features.emoji_analyzer",
        "apply_emoji",
        "Maps each emoji token to POSITIVE / NEGATIVE / NEUTRAL and injects\n"
        "     a sentiment placeholder (__EMOJI_POS__ etc.) into the sequence.",
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_import(module_path: str, callable_name: str):
    """
    Attempt to import *module_path* and return the named callable.
    Returns (callable, None) on success, (None, reason_str) on failure.
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None, f"module '{module_path}' not found (not implemented yet)"
    if not hasattr(mod, callable_name):
        return None, f"'{callable_name}' not defined in '{module_path}'"
    return getattr(mod, callable_name), None


def _status_line(level: int, label: str, active: bool, skip_reason: str = "") -> str:
    icon    = "✅" if active else ("⏳" if not skip_reason else "⚠️ ")
    status  = "ACTIVE" if active else ("SKIPPED — " + skip_reason if skip_reason else "DISABLED")
    return f"  [{level}] {icon}  {label:42s}  {status}"


def run_pipeline(tokens: list[str], max_level: int) -> dict:
    """
    Run the approach-2 pipeline up to *max_level*.
    Returns a dict with per-level results and metadata.
    """
    state   = list(tokens)   # mutable working copy
    results = {0: {"label": "Raw tokenised input", "tokens": list(state), "active": True}}

    for level in range(1, max_level + 1):
        label, mod_path, fn_name, _ = FEATURE_REGISTRY[level]
        fn, err = _try_import(mod_path, fn_name)

        if err:
            results[level] = {"label": label, "tokens": list(state),
                               "active": False, "skip_reason": err}
            continue

        try:
            state = fn(state)
        except Exception as exc:
            results[level] = {"label": label, "tokens": list(state),
                               "active": False, "skip_reason": f"runtime error: {exc}"}
            continue

        results[level] = {"label": label, "tokens": list(state), "active": True}

    return results


def print_banner(max_level: int):
    w = 70
    print("=" * w)
    print(" Approach 2 — Purely Statistical Hinglish Pipeline".center(w))
    print(f" Feature Demo  (running up to level {max_level})".center(w))
    print("=" * w)
    print()
    print("  FEATURE MAP")
    print("  " + "-" * 60)
    print(_status_line(0, "Baseline (raw tokens)", True))
    for level, (label, mod_path, fn_name, desc) in FEATURE_REGISTRY.items():
        fn, err = _try_import(mod_path, fn_name)
        active  = (level <= max_level) and (fn is not None)
        skip    = err if (level <= max_level and err) else ("" if level <= max_level else "level cap")
        print(_status_line(level, label, active, skip if not active else ""))
    print()


def print_trace(sentence: str, results: dict):
    print(f"  Input  : {sentence}")
    print("  " + "-" * 60)
    for level, info in results.items():
        status = "" if info["active"] else f"  [skipped: {info.get('skip_reason','')}]"
        print(f"  [{level}] {info['label']}")
        print(f"      → {info['tokens']}{status}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

DEMO_SENTENCES = [
    "yaar kya haal h bht accha lag rha hai 😊",
    "nhi nhi bhai yr serious nahi ho rha",
    "OMG sooo cute lol #Bollywood @RahulG",
    "kese aese mtlb nhi smjh pa rha tha",
    "thk hai bhai zarur aana kal",
]


def main():
    parser = argparse.ArgumentParser(
        description="Approach-2 feature-level demo for the Hinglish pipeline."
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=len(FEATURE_REGISTRY),
        choices=range(0, len(FEATURE_REGISTRY) + 1),
        help="Maximum feature level to activate (0 = baseline only, "
             f"{len(FEATURE_REGISTRY)} = all). Default: {len(FEATURE_REGISTRY)}",
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str,
        default=None,
        help="Custom sentence to process (in addition to demo sentences).",
    )
    args = parser.parse_args()

    tokenizer = _get_tokenizer()

    print_banner(args.level)

    sentences = DEMO_SENTENCES[:]
    if args.sentence:
        sentences = [args.sentence] + sentences

    for sentence in sentences:
        tokens  = tokenizer(sentence)
        results = run_pipeline(tokens, args.level)
        print_trace(sentence, results)


def _get_tokenizer():
    """
    Try to use Approach-2's own tokenizer; fall back to a simple whitespace
    split so the demo always works regardless of implementation state.
    """
    try:
        from src.approach1.features.tokenizer import HinglishTokenizer
        tok = HinglishTokenizer()
        return tok.tokenize
    except Exception:
        return str.split


if __name__ == "__main__":
    main()
