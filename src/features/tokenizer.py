"""
tokenizer.py
HinglishTokenizer: splits raw social-media text into clean tokens while
preserving hashtags, @mentions, and emojis as atomic units.
"""

import re
import emoji


class HinglishTokenizer:
    """
    Tokenizes raw Hinglish text from social media.

    Strategy
    --------
    1. Extract and protect hashtags, @mentions, and emojis as single tokens.
    2. Split on whitespace.
    3. For each non-protected token, separate leading/trailing punctuation
       and compress runs of the same punctuation character (e.g. "!!!" stays
       as one token "!!!").
    """

    # Patterns for protected spans
    _HASHTAG  = re.compile(r'#\w+')
    _MENTION  = re.compile(r'@\w+')
    _URL      = re.compile(r'https?://\S+|www\.\S+')

    # Informal repeated punctuation at end/start of a word
    _TRAIL_PUNCT = re.compile(r'([!?.,;:]+)$')
    _LEAD_PUNCT  = re.compile(r'^([!?.,;:]+)')

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Return a list of tokens for *text*."""
        tokens = []
        # Step 1 – replace protected spans with placeholders
        protected, text = self._protect(text)

        # Step 2 – whitespace split
        _ph_re = re.compile(r'^__PROTECTED_(\d+)__$')
        for raw_tok in text.split():
            m = _ph_re.match(raw_tok)
            if m:
                # restore original protected span
                idx = int(m.group(1))
                tokens.append(protected[idx])
            else:
                tokens.extend(self._split_punct(raw_tok))

        return [t for t in tokens if t.strip()]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _protect(self, text: str) -> tuple[list[str], str]:
        """
        Replace hashtags, @mentions, URLs, and emojis with numeric
        placeholders so they survive whitespace splitting intact.
        """
        protected: list[str] = []

        def _replace(m):
            protected.append(m.group(0))
            return f" __PROTECTED_{len(protected) - 1}__ "

        # Order matters: URLs before hashtags
        text = self._URL.sub(_replace, text)
        text = self._HASHTAG.sub(_replace, text)
        text = self._MENTION.sub(_replace, text)

        # Emoji demarcation
        result = []
        for char in text:
            if emoji.is_emoji(char):
                protected.append(char)
                result.append(f" __PROTECTED_{len(protected) - 1}__ ")
            else:
                result.append(char)
        text = "".join(result)

        return protected, text

    def _split_punct(self, token: str) -> list[str]:
        """
        Peel leading/trailing punctuation runs off a raw token.
        e.g.  'helloooo!!!'  →  ['helloooo', '!!!']
              '...wait'      →  ['...', 'wait']
        """
        parts = []

        # Leading punctuation
        m = self._LEAD_PUNCT.match(token)
        if m:
            parts.append(m.group(1))
            token = token[m.end():]

        # Trailing punctuation
        m = self._TRAIL_PUNCT.search(token)
        if m:
            parts.append(token[:m.start()])
            parts.append(m.group(1))
        else:
            parts.append(token)

        return [p for p in parts if p]


# ── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = HinglishTokenizer()
    samples = [
        "Yaar kya baat hai!!!😊 #Bollywood",
        "@RahulG bhai tum sahi bol rahe hoooo...",
        "OMG sooooo cute!! check https://example.com",
        "LOL iska kya matlab h?? bilkul nahi pata...",
    ]
    for s in samples:
        print(f"\nInput : {s}")
        print(f"Tokens: {tokenizer.tokenize(s)}")
