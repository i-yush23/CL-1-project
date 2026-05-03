"""
emoji_analyzer.py
Utilities for detecting emojis and mapping them to explicit emoji tags.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

# Broad emoji/symbol ranges commonly seen in chat data.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    # --- newly added ranges ---
    "\U0001F1E0-\U0001F1FF"  # regional indicator letters (flags)
    "\U0001F3F3-\U0001F3F4"  # white/black flag emoji (🏳 🏴)
    "\U0001F6A9"             # triangular flag (🚩)
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers (🏻🏼🏽🏾🏿)
    "\u0030-\u0039"          # digits 0-9 (base for keycap sequences)
    "\u0023"                 # # (hash keycap base)
    "\u002A"                 # * (asterisk keycap base)
    "]"
)

VARIATION_SELECTORS = {"\ufe0f", "\ufe0e"}
ZERO_WIDTH_JOINER = "\u200d"

# Keycap sequences: digit/# /* + variation selector fe0f + combining enclosing keycap 20e3
KEYCAP_RE = re.compile(
    r"[0-9#*]\ufe0f\u20e3"
)

# Flag sequences: two regional indicator letters forming a country flag
REGIONAL_FLAG_RE = re.compile(
    r"[\U0001F1E0-\U0001F1FF]{2}"
)


def _data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data", filename)


@lru_cache(maxsize=1)
def load_emoji_lexicon() -> dict[str, str]:
    path = _data_path("emoji_lexicon.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def is_emoji_token(token: str) -> bool:
    """Return True if the entire token is one emoji (including keycaps and flags)."""
    if not token:
        return False
    stripped = token.strip()
    if not stripped:
        return False

    # Keycap sequence check (e.g. 1️⃣)
    if KEYCAP_RE.fullmatch(stripped):
        return True

    # Country flag check (two regional indicator letters, e.g. 🇮🇳)
    if REGIONAL_FLAG_RE.fullmatch(stripped):
        return True

    cleaned = stripped.replace(ZERO_WIDTH_JOINER, "")
    for vs in VARIATION_SELECTORS:
        cleaned = cleaned.replace(vs, "")

    # Strip skin tone modifiers so base emoji still matches
    cleaned = re.sub(r"[\U0001F3FB-\U0001F3FF]", "", cleaned)

    return bool(cleaned) and all(EMOJI_RE.fullmatch(ch) for ch in cleaned)


def extract_emojis(text: str) -> list[str]:
    """Extract all emoji tokens from text, including keycaps and flags."""
    results: list[str] = []

    # Keycap sequences first (consume before single-char regex)
    text_remaining = text
    for m in KEYCAP_RE.finditer(text):
        results.append(m.group())

    # Country flags (two regional indicator letters)
    for m in REGIONAL_FLAG_RE.finditer(text):
        results.append(m.group())

    # Standard single-codepoint emojis
    results.extend(EMOJI_RE.findall(text))

    return results


def emoji_placeholder(token: str) -> str:
    """Map an emoji token to a tagged placeholder like __EMOJI_POSITIVE__."""
    sentiment = load_emoji_lexicon().get(token, "NEUTRAL")
    return f"__EMOJI_{sentiment}__"
