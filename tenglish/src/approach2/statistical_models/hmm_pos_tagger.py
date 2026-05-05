import re
"""
hmm_pos_tagger.py  (improved)
HMM-based Part-of-Speech tagger for Tenglish (Telugu-English code-mixed) text.

Key improvements vs original:
  1. Laplace (add-1) smoothing in training so zero-count tags still get non-zero
     emission probabilities — reduces over-confidence on seen words.
  2. Suffix heuristics for Telugu verb forms (-adu, -indi, -tunna*, -tadu, -du)
     and noun/adjective suffixes (-tion, -ness, -ly, -ing as fallback signals).
  3. Expanded heuristic word-lists: more Telugu pronouns, adpositions, particles,
     common slang, more English function words.
  4. Unknown-word morphological fallback: when phonetic hash also misses,
     use suffix-based tag prior rather than flat smoothing.
  5. Laplace-smoothed transitions (add-alpha) to handle unseen tag bigrams.
  6. Model save/load now uses the script's own directory as default, so the
     tagger works regardless of the working directory.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from approach2.features.emoji_analyzer import is_emoji_token
from approach2.features.phonetic_matcher import get_phonetic_hash

# Default data directory relative to this file (works from any cwd)
_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


class HMMPosTagger:

    def __init__(self):
        self.emission_probs          = defaultdict(lambda: defaultdict(float))
        self.phonetic_emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs        = defaultdict(lambda: defaultdict(float))
        self.unigram_probs           = defaultdict(lambda: defaultdict(int))
        self.tag_counts              = defaultdict(int)
        self.all_tags: list[str]     = []
        self.seen_words: set[str]    = set()
        self.vocab_size: int         = 0   # for Laplace smoothing

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_path: str = "data/raw/train_pos.csv"):
        """
        Train from a CSV with columns: token, label
        Blank token rows are treated as sentence boundaries.
        """
        tag_word_counts       = defaultdict(lambda: defaultdict(int))
        tag_phonetic_counts   = defaultdict(lambda: defaultdict(int))
        tag_transition_counts = defaultdict(lambda: defaultdict(int))

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                reader   = csv.DictReader(f)
                prev_tag = "START"

                for row in reader:
                    word = row["token"].strip().lower()
                    tag  = row["label"].strip()

                    if not word or not tag:
                        prev_tag = "START"
                        continue

                    tag_word_counts[tag][word] += 1
                    self.seen_words.add(word)

                    ph = get_phonetic_hash(word)
                    tag_phonetic_counts[tag][ph] += 1

                    self.unigram_probs[word][tag] += 1
                    self.tag_counts[tag] += 1
                    tag_transition_counts[prev_tag][tag] += 1
                    prev_tag = tag

        except FileNotFoundError:
            print(f"[ERROR] Training file not found: {data_path}")
            return

        self.all_tags    = list(self.tag_counts.keys())
        self.vocab_size  = len(self.seen_words)
        num_tags         = len(self.all_tags)

        # Laplace (add-1) smoothed emission probabilities
        for tag in self.all_tags:
            total = self.tag_counts[tag]
            for w, count in tag_word_counts[tag].items():
                self.emission_probs[tag][w] = (count + 1) / (total + self.vocab_size)

        for tag in self.all_tags:
            total = self.tag_counts[tag]
            for ph, count in tag_phonetic_counts[tag].items():
                self.phonetic_emission_probs[tag][ph] = (count + 1) / (total + self.vocab_size)

        # Add-alpha smoothed transition probabilities (alpha=0.1)
        alpha = 0.1
        all_prev_tags = list(tag_transition_counts.keys())
        for p_tag in all_prev_tags:
            total = sum(tag_transition_counts[p_tag].values())
            for n_tag in self.all_tags:
                count = tag_transition_counts[p_tag].get(n_tag, 0)
                self.transition_probs[p_tag][n_tag] = (count + alpha) / (total + alpha * num_tags)

        print(f"[HMM] Trained on {len(self.all_tags)} unique POS tags, "
              f"{self.vocab_size} unique words.")

        # Persist model
        os.makedirs(_DEFAULT_DATA_DIR, exist_ok=True)
        with open(os.path.join(_DEFAULT_DATA_DIR, "hmm_emissions.json"),          "w") as f:
            json.dump(dict(self.emission_probs), f)
        with open(os.path.join(_DEFAULT_DATA_DIR, "hmm_phonetic_emissions.json"), "w") as f:
            json.dump(dict(self.phonetic_emission_probs), f)
        with open(os.path.join(_DEFAULT_DATA_DIR, "hmm_transitions.json"),        "w") as f:
            json.dump(dict(self.transition_probs), f)
        with open(os.path.join(_DEFAULT_DATA_DIR, "hmm_unigrams.json"),           "w") as f:
            json.dump(dict(self.unigram_probs), f)
        print(f"[HMM] Model files saved to {_DEFAULT_DATA_DIR}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, data_dir: str = _DEFAULT_DATA_DIR):
        try:
            with open(os.path.join(data_dir, "hmm_emissions.json"),          "r") as f:
                self.emission_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_phonetic_emissions.json"), "r") as f:
                self.phonetic_emission_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_transitions.json"),        "r") as f:
                self.transition_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_unigrams.json"),           "r") as f:
                self.unigram_probs = json.load(f)

            self.all_tags   = list(self.emission_probs.keys())
            self.seen_words = set()
            for tags in self.emission_probs.values():
                self.seen_words.update(tags.keys())
            self.vocab_size = len(self.seen_words)
            print(f"[HMM] Loaded model: {len(self.all_tags)} tags, "
                  f"{len(self.seen_words)} known words.")
        except FileNotFoundError as e:
            print(f"[ERROR] Model file missing: {e}. Train the model first.")

    # ------------------------------------------------------------------
    # Telugu/English suffix heuristics for unknown words
    # ------------------------------------------------------------------

    _TELUGU_VERB_SUFFIXES = (
        "tunna", "tundi", "tadu", "dadu", "indi", "indi",
        "adu", "edu", "inchu", "ించు", "staa", "stav",
        "leka", "ledhu", "ledu",
    )
    _TELUGU_NOUN_SUFFIXES = ("pu", "vu", "mu", "lu", "gaa", "ki", "ku")
    _ENGLISH_VERB_SUFFIXES = ("ing", "tion", "ed", "ify", "ise", "ize")
    _ENGLISH_ADJ_SUFFIXES  = ("ful", "less", "ous", "ive", "ic", "able", "ible")
    _ENGLISH_ADV_SUFFIXES  = ("ly",)
    _ENGLISH_NOUN_SUFFIXES = ("tion", "sion", "ness", "ment", "ity", "er", "or")

    def _suffix_tag_prior(self, word: str) -> str | None:
        """Return a tag hint based on word suffix, or None if unclear."""
        w = word.lower()
        for suf in self._TELUGU_VERB_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return "V"
        for suf in self._TELUGU_NOUN_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return "N"
        for suf in self._ENGLISH_VERB_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return "V"
        for suf in self._ENGLISH_ADJ_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return "ADJ"
        for suf in self._ENGLISH_ADV_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return "ADV"
        for suf in self._ENGLISH_NOUN_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return "N"
        return None

    # ------------------------------------------------------------------
    # Heuristic rule-based tag assignment
    # ------------------------------------------------------------------

    def _looks_like_number(self, word: str) -> bool:
        return any(ch.isdigit() for ch in word)

    def _heuristic_tag(self, word: str) -> str | None:
        low = word.lower()
        punct = {".", ",", "!", "?", ";", ":", "(", ")", "[", "]",
                 "{", "}", "--", "...", "—", "…",
                 # multi-char punct common in Tenglish social media
                 "..", "....", ".....", "??", "???", "????",
                 "!!", "!!!", "!?", "?!", ":)", ":(", ":D", ":/",
                 "xd", "xD", ">", "<", "/", "\\", "|", "~",
                 "\\m/", "\\m\\", "*", "^", "="}
        if is_emoji_token(word):
            return "EMOJI"
        if low in punct:
            return "PUNC"
        if low.startswith(("http", "www.")) or low.startswith(("@", "#")):
            return "X"
        # URLs without http, pic.twitter.com, t.co etc
        if re.search(r"\.(com|org|net|co|ly|in|io)/", low):
            return "X"
        # Pure punctuation sequences not caught above
        if re.match(r"^[^a-z0-9]+$", low) and len(low) >= 2:
            return "PUNC"
        # Emoticons and symbol sequences
        if re.match(r"^[\\\\/*|~^=<>]+$", low):
            return "PUNC"
        if self._looks_like_number(low):
            return "NUM"

        pronouns = {
            # English
            "i", "you", "he", "she", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "our", "their", "its", "it",
            # Telugu
            "nenu", "nuvvu", "meeru", "memu", "manam", "vaadu", "aame",
            "adi", "idi", "naaku", "maaku", "meeku", "vaallu", "vaallaki",
            "tantam", "tanu", "tanaki", "evaru", "emiti", "emi", "ela",
            "ekkada", "eppudu", "vaallu", "mee", "naa", "ninnu", "nannu",
        }
        determiners = {
            "a", "an", "the", "this", "that", "these", "those",
            "oka", "aa", "ee", "emi",
        }
        conjunctions = {
            "and", "or", "but", "because", "if", "though", "although",
            "so", "yet", "nor", "either", "neither", "while", "since",
            "kani", "kaani", "ledu", "ayina", "ante", "mariyu", "ledha",
            "inkaa", "kaabatti", "alage", "anduvalla",
        }
        adpositions = {
            "in", "on", "at", "to", "from", "with", "for", "of", "by",
            "into", "onto", "over", "under", "about", "between",
            # Telugu postpositions
            "meeda", "ki", "ku", "lo", "tho", "nunchi", "varaku",
            "gurinchi", "kosam", "mundhu", "venaka", "dwara", "valla",
            "tarvata", "mundu", "lopala", "bayata", "daggara",
        }
        particles = {
            "not", "na", "kada", "emo", "ante", "ani", "okka",
            "kooda", "kuda", "emiti", "enti",
        }
        adjectives = {
            "good", "bad", "great", "nice", "happy", "sad", "big",
            "small", "new", "old", "hot", "cold", "fast", "slow",
            "chaala", "chala", "bayya", "bagundi", "super",
        }
        adverbs = {
            "very", "really", "too", "just", "only", "also", "why",
            "how", "when", "where", "ippudu", "appudu", "inka",
            "already", "still", "yet", "soon", "always", "never",
        }
        verbs = {
            "do", "does", "did", "know", "go", "come", "have", "has",
            "had", "be", "is", "am", "are", "was", "were", "will",
            "would", "could", "should", "may", "might", "can",
            "get", "got", "give", "take", "make", "see", "say",
            "want", "need", "like", "love",
        }
        nouns = {
            "brother", "picture", "photo", "message", "situation",
            "friend", "time", "day", "people", "thing", "place",
            "money", "work", "life", "home", "family",
        }
        interj = {
            "hey", "arre", "arey", "ayyo", "abba", "wow", "haha",
            "hahaha", "lol", "oh", "oops", "yes", "no", "ok",
            "okay", "hmm", "ah", "ugh",
        }
        propn_hints = {
            "hyderabad", "bangalore", "chennai", "mumbai", "delhi",
            "telugu", "india", "google", "whatsapp", "instagram",
        }

        if low in pronouns:    return "PRON"
        if low in determiners: return "DET"
        if low in conjunctions: return "CONJ"
        if low in adpositions:  return "ADP"
        if low in particles:    return "PART"
        if low in adjectives:   return "ADJ"
        if low in adverbs:      return "ADV"
        if low in verbs:        return "V"
        if low in nouns:        return "N"
        if low in interj:       return "INTJ"
        if low in propn_hints:  return "PROPN"
        return None

    # ------------------------------------------------------------------
    # Emission probability with Laplace fallback + suffix prior
    # ------------------------------------------------------------------

    def _get_emission_prob(self, tag: str, word: str, smoothing: float) -> float:
        forced = self._heuristic_tag(word)
        if forced is not None:
            return 1.0 if tag == forced else smoothing

        low = word.lower()

        # Unigram backoff (most reliable for seen words)
        if low in self.unigram_probs and tag in self.unigram_probs[low]:
            total = sum(self.unigram_probs[low].values())
            if total > 0:
                return self.unigram_probs[low][tag] / total

        # Unseen word: try phonetic hash
        if low not in self.seen_words:
            ph = get_phonetic_hash(low)
            phonetic_tag_probs = self.phonetic_emission_probs.get(tag, {})
            if ph in phonetic_tag_probs:
                return phonetic_tag_probs[ph]

            # Suffix heuristic prior: boost the likely tag
            suffix_tag = self._suffix_tag_prior(low)
            if suffix_tag is not None:
                if tag == suffix_tag:
                    return 1e-4   # above smoothing but below seen-word probs
                return smoothing

            return smoothing

        # Known word, standard emission
        return self.emission_probs.get(tag, {}).get(low, smoothing)

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def viterbi_decode(self, tokens: list[str]) -> list[tuple[str, str]]:
        if not tokens:
            return []

        SMOOTHING = 1e-10

        viterbi:      list[dict] = []
        backpointers: list[dict] = []

        # ── Initialisation ────────────────────────────────────────────
        first_col: dict[str, float] = {}
        forced_first = self._heuristic_tag(tokens[0])
        if forced_first in {"PUNC", "EMOJI"}:
            for tag in self.all_tags + ([forced_first] if forced_first not in self.all_tags else []):
                first_col[tag] = -1e15
            first_col[forced_first] = 0.0
        else:
            for tag in self.all_tags:
                t_prob = self.transition_probs.get("START", {}).get(tag, SMOOTHING)
                e_prob = self._get_emission_prob(tag, tokens[0].lower(), SMOOTHING)
                first_col[tag] = math.log(t_prob) + math.log(e_prob)

        viterbi.append(first_col)

        # ── Recursion ─────────────────────────────────────────────────
        extra_tags = []
        if "PUNC"  not in self.all_tags: extra_tags.append("PUNC")
        if "EMOJI" not in self.all_tags: extra_tags.append("EMOJI")
        all_tags_ext = self.all_tags + extra_tags

        for i in range(1, len(tokens)):
            curr_col: dict[str, float] = {}
            curr_bp:  dict[str, str]   = {}
            word         = tokens[i].lower()
            forced_tag   = self._heuristic_tag(tokens[i])

            for curr_tag in all_tags_ext:

                # Hard override: PUNC
                if forced_tag == "PUNC":
                    if curr_tag == "PUNC":
                        best_prev, max_p = self.all_tags[0], -1e15
                        for pt in self.all_tags:
                            lp = (viterbi[i-1].get(pt, -1e15)
                                  + math.log(self.transition_probs.get(pt, {}).get("PUNC", SMOOTHING)))
                            if lp > max_p:
                                max_p, best_prev = lp, pt
                        curr_col["PUNC"] = max_p
                        curr_bp["PUNC"]  = best_prev
                    else:
                        curr_col[curr_tag] = -1e15
                    continue

                # Hard override: EMOJI
                if forced_tag == "EMOJI":
                    if curr_tag == "EMOJI":
                        best_prev, max_p = self.all_tags[0], -1e15
                        for pt in self.all_tags:
                            lp = (viterbi[i-1].get(pt, -1e15)
                                  + math.log(self.transition_probs.get(pt, {}).get("EMOJI", SMOOTHING)))
                            if lp > max_p:
                                max_p, best_prev = lp, pt
                        curr_col["EMOJI"] = max_p
                        curr_bp["EMOJI"]  = best_prev
                    else:
                        curr_col[curr_tag] = -1e15
                    continue

                # Skip PUNC/EMOJI slots for non-special tokens
                if curr_tag in {"PUNC", "EMOJI"}:
                    curr_col[curr_tag] = -1e15
                    continue

                e_prob     = self._get_emission_prob(curr_tag, word, SMOOTHING)
                log_e_prob = math.log(e_prob)

                best_prev = self.all_tags[0]
                best_lp   = (viterbi[i-1].get(best_prev, -1e15)
                             + math.log(self.transition_probs.get(best_prev, {}).get(curr_tag, SMOOTHING))
                             + log_e_prob)
                for prev_tag in self.all_tags[1:]:
                    lp = (viterbi[i-1].get(prev_tag, -1e15)
                          + math.log(self.transition_probs.get(prev_tag, {}).get(curr_tag, SMOOTHING))
                          + log_e_prob)
                    if lp > best_lp:
                        best_lp, best_prev = lp, prev_tag

                curr_col[curr_tag] = best_lp
                curr_bp[curr_tag]  = best_prev

            viterbi.append(curr_col)
            backpointers.append(curr_bp)

        # ── Termination ───────────────────────────────────────────────
        best_last = max(viterbi[-1], key=viterbi[-1].get)
        path = [best_last]
        for i in range(len(backpointers) - 1, -1, -1):
            best_last = backpointers[i].get(best_last, self.all_tags[0])
            path.append(best_last)
        path.reverse()

        # Final hard override for punctuation/emoji
        results = []
        for tok, tag in zip(tokens, path):
            forced = self._heuristic_tag(tok)
            results.append((tok, forced if forced is not None else tag))
        return results


# ---------------------------------------------------------------------------
# CLI demo / training entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tagger = HMMPosTagger()

    # Resolve train CSV relative to project root (two levels up from this file)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    train_csv = os.path.join(_project_root, "data", "raw", "train_pos.csv")

    if os.path.exists(train_csv):
        tagger.train(train_csv)
    else:
        print(f"[INFO] No training file found at {train_csv}. Loading pre-trained model.")
        tagger.load()

    if tagger.all_tags:
        test = ["nuvvu", "chestunnav", "em", "?", "I", "have", "no", "idea", "bro", "😂"]
        print(f"\n{'Token':<20} {'POS'}")
        print("-" * 28)
        for tok, tag in tagger.viterbi_decode(test):
            print(f"{tok:<20} {tag}")
