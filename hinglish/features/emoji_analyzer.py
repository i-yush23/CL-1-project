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
    "]"
)

VARIATION_SELECTORS = {"\ufe0f", "\ufe0e"}
ZERO_WIDTH_JOINER = "\u200d"


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
    if not token:
        return False
    stripped = token.strip()
    if not stripped:
        return False
    cleaned = stripped.replace(ZERO_WIDTH_JOINER, "")
    for vs in VARIATION_SELECTORS:
        cleaned = cleaned.replace(vs, "")
    return bool(cleaned) and all(EMOJI_RE.fullmatch(ch) for ch in cleaned)


def extract_emojis(text: str) -> list[str]:
    return EMOJI_RE.findall(text)


def emoji_placeholder(token: str) -> str:
    sentiment = load_emoji_lexicon().get(token, "NEUTRAL")
    return f"__EMOJI_{sentiment}__"
