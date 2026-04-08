import json
import csv
import math
import os
import sys
from collections import defaultdict

# Add src to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from approach2.features.phonetic_matcher import get_phonetic_hash

class HMMPosTagger:
    def __init__(self):
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.phonetic_emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.unigram_probs = defaultdict(lambda: defaultdict(int)) # word -> {tag: count}
        self.tag_counts = defaultdict(int)
        self.all_tags = []
        self.seen_words = set()

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
        os.makedirs("src/approach2/data", exist_ok=True)
        with open("src/approach2/data/hmm_emissions.json", 'w') as f:
            json.dump(self.emission_probs, f)
        with open("src/approach2/data/hmm_phonetic_emissions.json", 'w') as f:
            json.dump(self.phonetic_emission_probs, f)
        with open("src/approach2/data/hmm_transitions.json", 'w') as f:
            json.dump(self.transition_probs, f)
        with open("src/approach2/data/hmm_unigrams.json", 'w') as f:
            json.dump(self.unigram_probs, f)
        
    def load(self, data_dir="src/approach2/data"):
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
        except FileNotFoundError:
            print("Error: Model files not found. Please train the model first.")

    def _get_emission_prob(self, tag, word, smoothing):
        """Helper for emission logic with Phonetic Folding fallback."""
        low_word = word.lower()
        
        # 1. Word is completely unknown to the corpus -> Try phonetic folding
        if low_word not in self.seen_words:
            p_hash = get_phonetic_hash(low_word)
            if p_hash in self.phonetic_emission_probs[tag]:
                # If phonetic variations are common for this tag, return phonetic prob
                return self.phonetic_emission_probs[tag].get(p_hash, smoothing)
            return smoothing

        # 2. Word is known in the corpus -> Use standard emission prob
        # (If it's known but not seen with THIS tag, return smoothing)
        return self.emission_probs[tag].get(low_word, smoothing)

    def viterbi_decode(self, tokens):
        if not tokens:
            return []

        # Punctuation list
        PUNCTUATIONS = set([".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "--", "..."])
        
        SMOOTHING = 1e-10
        viterbi = []
        backpointers = []

        # Initialization
        first_col = {}
        first_word = tokens[0].lower()
        
        # Hard override for punctuation in initialization
        if first_word in PUNCTUATIONS:
            for tag in self.all_tags:
                first_col[tag] = -1e15 # very low
            if "PUNC" not in first_col: first_col["PUNC"] = 0.0
            else: first_col["PUNC"] = 0.0
        else:
            for tag in self.all_tags:
                t_prob = self.transition_probs.get("START", {}).get(tag, SMOOTHING)
                e_prob = self._get_emission_prob(tag, first_word, SMOOTHING)
                first_col[tag] = math.log(t_prob) + math.log(e_prob)
        viterbi.append(first_col)

        # Recursion
        for i in range(1, len(tokens)):
            curr_col = {}
            curr_backpointers = {}
            word = tokens[i].lower()

            for curr_tag in self.all_tags + (["PUNC"] if "PUNC" not in self.all_tags else []):
                # Hard override for punctuation tokens
                if word in PUNCTUATIONS:
                    if curr_tag == "PUNC":
                        # Find best previous tag that leads to PUNC
                        best_prev = self.all_tags[0]
                        max_p = viterbi[i-1][best_prev] + math.log(self.transition_probs.get(best_prev, {}).get("PUNC", SMOOTHING))
                        for pt in self.all_tags[1:]:
                            lp = viterbi[i-1][pt] + math.log(self.transition_probs.get(pt, {}).get("PUNC", SMOOTHING))
                            if lp > max_p:
                                max_p = lp
                                best_prev = pt
                        curr_col["PUNC"] = max_p
                        curr_backpointers["PUNC"] = best_prev
                    else:
                        curr_col[curr_tag] = -1e15
                else:
                    # Standard Viterbi
                    if curr_tag == "PUNC":
                        curr_col["PUNC"] = -1e15
                        continue
                        
                    e_prob = self._get_emission_prob(curr_tag, word, SMOOTHING)
                    log_e_prob = math.log(e_prob)
                    
                    best_prev_tag = self.all_tags[0]
                    t_prob = self.transition_probs.get(best_prev_tag, {}).get(curr_tag, SMOOTHING)
                    max_log_prob = viterbi[i-1][best_prev_tag] + math.log(t_prob) + log_e_prob

                    for prev_tag in self.all_tags[1:]:
                        if prev_tag not in viterbi[i-1]: continue
                        t_prob = self.transition_probs.get(prev_tag, {}).get(curr_tag, SMOOTHING)
                        log_prob = viterbi[i-1][prev_tag] + math.log(t_prob) + log_e_prob
                        if log_prob > max_log_prob:
                            max_log_prob = log_prob
                            best_prev_tag = prev_tag
                    
                    curr_col[curr_tag] = max_log_prob
                    curr_backpointers[curr_tag] = best_prev_tag
            
            viterbi.append(curr_col)
            backpointers.append(curr_backpointers)

        # Termination
        best_last_tag = max(viterbi[-1], key=viterbi[-1].get)
        path = [best_last_tag]
        for i in range(len(backpointers) - 1, -1, -1):
            best_last_tag = backpointers[i].get(best_last_tag, self.all_tags[0])
            path.append(best_last_tag)
        
        path.reverse()
        # FINAL Pass to ensure all PUNCS are tagged PUNC in results
        results = []
        for tok, tag in zip(tokens, path):
            if tok.lower() in PUNCTUATIONS:
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


        