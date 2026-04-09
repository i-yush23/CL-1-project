"""
pipeline_runner.py
End-to-end Tenglish (Telugu-English) NLP pipeline.

Pipeline:
  Raw text
    → tokenize (words + punctuation + emojis)
    → slang expansion / normalization
    → CodeswitchSegmenter  →  LID labels  (TE / EN / UNI / EMOJI)
    → HMMPosTagger         →  POS tags (including EMOJI)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from approach2.features.emoji_analyzer import is_emoji_token
from approach2.statistical_models.code_switch_labeler import CodeswitchSegmenter
from approach2.statistical_models.hmm_pos_tagger import HMMPosTagger


def _data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def load_slang_map() -> dict[str, str]:
    path = _data_path("slang_map.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def tokenize_text(text: str) -> list[str]:
    """Tokenize while keeping emojis as standalone tokens."""
    tokens: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    keep_inside_word = {"_", "'", "-", "@", "#", "/", ":", "."}

    for ch in text:
        if ch.isspace():
            flush_current()
            continue
        if is_emoji_token(ch):
            flush_current()
            tokens.append(ch)
            continue
        if ch.isalnum() or ch in keep_inside_word:
            current.append(ch)
            continue
        flush_current()
        tokens.append(ch)

    flush_current()
    return [tok for tok in tokens if tok]


def expand_slang(tokens: Iterable[str], slang_map: dict[str, str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        key = token.lower()
        if key in slang_map:
            expanded.extend(slang_map[key].split())
        else:
            expanded.append(token)
    return expanded


def run_pipeline(text: str) -> None:
    segmenter = CodeswitchSegmenter()
    tagger = HMMPosTagger()
    tagger.load()

    raw_tokens = tokenize_text(text)
    tokens = expand_slang(raw_tokens, load_slang_map())

    if not tokens:
        print("[WARN] No tokens found in input.")
        return

    lid_labels = segmenter.segment(tokens)
    pos_results = tagger.viterbi_decode(tokens)

    print(f"\n{'Token':<20} | {'LID':<6} | {'POS'}")
    print("-" * 42)
    for token, lid, (_, pos) in zip(tokens, lid_labels, pos_results):
        print(f"{token:<20} | {lid:<6} | {pos}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = "Nuvvu ikkade undi 😂 but idk why bro, naaku scene ardham kaledu!"

    run_pipeline(test_text)
