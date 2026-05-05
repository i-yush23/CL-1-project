import json
import csv
import math
import os
import sys
from collections import defaultdict

# Add src to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hinglish.features.phonetic_matcher import get_phonetic_hash

# Tags that should NEVER be assigned to a purely alphabetic word token
_WORD_EXCLUDED_TAGS = frozenset({"SYM", "PUNC", "$", "null", "en", "E"})

class HMMPosTagger:
    def __init__(self):
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.phonetic_emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.unigram_probs = defaultdict(lambda: defaultdict(int)) # word -> {tag: count}
        self.tag_counts = defaultdict(int)
        self.all_tags = []
        self.seen_words = set()
        self._tag_prior = {}  # tag -> P(tag) from corpus frequencies

    def train(self, data_path="data/raw/train_pos.csv"):
        tag_word_counts = defaultdict(lambda: defaultdict(int))
        tag_phonetic_counts = defaultdict(lambda: defaultdict(int))
        tag_transition_counts = defaultdict(lambda: defaultdict(int))
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                prev_tag = "START"

                for row in reader:
                    word = row['token'].strip().lower()
                    tag = row['label'].strip()

                    if not word or not tag:
                        prev_tag = "START"
                        continue

                    # Standard emission
                    tag_word_counts[tag][word] += 1
                    self.seen_words.add(word)
                    
                    # Phonetic emission (folding)
                    p_hash = get_phonetic_hash(word)
                    tag_phonetic_counts[tag][p_hash] += 1
                    
                    # Unigram backoff data
                    self.unigram_probs[word][tag] += 1
                    
                    # Transition counts
                    self.tag_counts[tag] += 1
                    tag_transition_counts[prev_tag][tag] += 1
                    prev_tag = tag
        except FileNotFoundError:
            print(f"Error: Training file not found at {data_path}")
            return

        # Calculate standard emission probabilities
        for tag, words in tag_word_counts.items():
            for w, count in words.items():
                self.emission_probs[tag][w] = count / self.tag_counts[tag]

        # Calculate phonetic emission probabilities
        for tag, phashes in tag_phonetic_counts.items():
            for ph, count in phashes.items():
                self.phonetic_emission_probs[tag][ph] = count / self.tag_counts[tag]

        # Calculate transition probabilities
        for p_tag, n_tags in tag_transition_counts.items():
            total_transitions = sum(n_tags.values())
            for n_tag, count in n_tags.items():
                self.transition_probs[p_tag][n_tag] = count / total_transitions

        self.all_tags = list(self.tag_counts.keys())
        print(f"Trained on {len(self.all_tags)} unique tags")

        # Save models
        os.makedirs("hinglish/data", exist_ok=True)
        with open("hinglish/data/hmm_emissions.json", 'w') as f:
            json.dump(self.emission_probs, f)
        with open("hinglish/data/hmm_phonetic_emissions.json", 'w') as f:
            json.dump(self.phonetic_emission_probs, f)
        with open("hinglish/data/hmm_transitions.json", 'w') as f:
            json.dump(self.transition_probs, f)
        with open("hinglish/data/hmm_unigrams.json", 'w') as f:
            json.dump(self.unigram_probs, f)
        
    def load(self, data_dir="hinglish/data"):
        try:
            with open(os.path.join(data_dir, "hmm_emissions.json"), 'r') as f:
                self.emission_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_phonetic_emissions.json"), 'r') as f:
                self.phonetic_emission_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_transitions.json"), 'r') as f:
                self.transition_probs = json.load(f)
            with open(os.path.join(data_dir, "hmm_unigrams.json"), 'r') as f:
                self.unigram_probs = json.load(f)
            
            self.all_tags = list(self.emission_probs.keys())
            self.seen_words = set()
            for tags in self.emission_probs.values():
                self.seen_words.update(tags.keys())

            # Build a tag frequency prior so unknown-word fallback is
            # proportional to how common each tag is in the corpus.
            # This prevents the Viterbi drifting toward dominant-transition
            # tags (like SYM) for out-of-vocabulary words.
            total_tag_tokens = sum(
                sum(word_tags.values())
                for word_tags in self.unigram_probs.values()
            )
            tag_totals: dict[str, int] = defaultdict(int)
            for word_tags in self.unigram_probs.values():
                for tag, cnt in word_tags.items():
                    tag_totals[tag] += cnt
            if total_tag_tokens > 0:
                self._tag_prior = {
                    tag: cnt / total_tag_tokens
                    for tag, cnt in tag_totals.items()
                }
            else:
                self._tag_prior = {}
        except FileNotFoundError:
            print("Error: Model files not found. Please train the model first.")

    def _get_emission_prob(self, tag, word, smoothing, is_alpha_word=True):
        """Helper for emission logic with Phonetic Folding fallback.

        For purely alphabetic tokens, SYM/PUNC tags get a near-zero
        probability so Viterbi never picks them for real words.
        """
        low_word = word.lower()

        # Never emit a symbol/punctuation tag for a real alphabetic word
        if is_alpha_word and tag in _WORD_EXCLUDED_TAGS:
            return smoothing

        # 1. Word is completely unknown to the corpus -> Try phonetic folding
        if low_word not in self.seen_words:
            p_hash = get_phonetic_hash(low_word)
            if p_hash in self.phonetic_emission_probs.get(tag, {}):
                return self.phonetic_emission_probs[tag][p_hash]
            # Fall back to tag-frequency prior so well-attested tags
            # (N, V, PSP, …) win over rare/noisy ones for unknown words.
            return self._tag_prior.get(tag, smoothing)

        # 2. Word is known in the corpus -> Use standard emission prob
        return self.emission_probs.get(tag, {}).get(low_word, smoothing)

    def viterbi_decode(self, tokens):
        if not tokens:
            return []

        PUNCTUATIONS = set(["." , ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "--", "..."])
        SMOOTHING = 1e-10
        viterbi = []
        backpointers = []

        all_tags_with_punc = self.all_tags + (["PUNC"] if "PUNC" not in self.all_tags else [])

        # ── helpers ──────────────────────────────────────────────────────────
        def _is_alpha(tok):
            """True if the token (after stripping apostrophes) is purely alphabetic."""
            return tok.replace("'", "").isalpha()

        def _candidate_tags(tok):
            """Return the set of tags to consider for this token."""
            if tok in PUNCTUATIONS:
                return ["PUNC"]
            if _is_alpha(tok):
                # Exclude symbol-class tags for real words
                return [t for t in all_tags_with_punc if t not in _WORD_EXCLUDED_TAGS]
            return all_tags_with_punc

        # ── Initialization ───────────────────────────────────────────────────
        first_col = {}
        first_word = tokens[0].lower()
        first_is_alpha = _is_alpha(tokens[0])

        if tokens[0] in PUNCTUATIONS:
            for tag in all_tags_with_punc:
                first_col[tag] = -1e15
            first_col["PUNC"] = 0.0
        else:
            for tag in _candidate_tags(tokens[0]):
                t_prob = self.transition_probs.get("START", {}).get(tag, SMOOTHING)
                e_prob = self._get_emission_prob(tag, first_word, SMOOTHING, first_is_alpha)
                first_col[tag] = math.log(t_prob) + math.log(e_prob)
        viterbi.append(first_col)

        # ── Recursion ─────────────────────────────────────────────────────────
        for i in range(1, len(tokens)):
            curr_col = {}
            curr_backpointers = {}
            word = tokens[i].lower()
            tok = tokens[i]
            is_alpha = _is_alpha(tok)
            candidates = _candidate_tags(tok)

            for curr_tag in all_tags_with_punc:
                if curr_tag not in candidates:
                    curr_col[curr_tag] = -1e15
                    continue

                if tok in PUNCTUATIONS:
                    # PUNC: find best predecessor
                    best_prev, max_p = None, -math.inf
                    for pt, prev_score in viterbi[i-1].items():
                        p = prev_score + math.log(
                            self.transition_probs.get(pt, {}).get("PUNC", SMOOTHING)
                        )
                        if p > max_p:
                            max_p, best_prev = p, pt
                    curr_col["PUNC"] = max_p
                    curr_backpointers["PUNC"] = best_prev
                else:
                    e_prob = self._get_emission_prob(curr_tag, word, SMOOTHING, is_alpha)
                    log_e = math.log(e_prob)

                    best_prev, max_lp = None, -math.inf
                    for prev_tag, prev_score in viterbi[i-1].items():
                        t_prob = self.transition_probs.get(prev_tag, {}).get(curr_tag, SMOOTHING)
                        lp = prev_score + math.log(t_prob) + log_e
                        if lp > max_lp:
                            max_lp, best_prev = lp, prev_tag

                    curr_col[curr_tag] = max_lp
                    curr_backpointers[curr_tag] = best_prev

            viterbi.append(curr_col)
            backpointers.append(curr_backpointers)

        # ── Termination ───────────────────────────────────────────────────────
        best_last_tag = max(viterbi[-1], key=viterbi[-1].get)
        path = [best_last_tag]
        for i in range(len(backpointers) - 1, -1, -1):
            best_last_tag = backpointers[i].get(best_last_tag, self.all_tags[0])
            path.append(best_last_tag)

        path.reverse()
        # Final pass: hard-enforce PUNC for punctuation characters
        results = []
        for tok, tag in zip(tokens, path):
            if tok in PUNCTUATIONS:
                results.append((tok, "PUNC"))
            else:
                results.append((tok, tag))
        return results

if __name__ == "__main__":
    tagger = HMMPosTagger()
    train_data = "data/raw/train_pos.csv"
    if os.path.exists(train_data):
        tagger.train(train_data)
    
    tagger.load()
    if tagger.all_tags:
        test_tokens = ["the", "boy", "wanted", "to", "watch", "a", "mela"]
        result = tagger.viterbi_decode(test_tokens)
        for token, tag in result:
            print(f"{token:15} {tag}")


        