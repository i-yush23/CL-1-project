"""
lid_model.py — Statistical Language Identification model for the Hinglish pipeline.

Uses a Naive Bayes emission model (word → language) trained on the BIS LID corpus
(labels: EN / TE / UNI), combined with a sliding-window context smoother.

Training:
    python hinglish/statistical_models/lid_model.py --train

Inference:
    from hinglish.statistical_models.lid_model import LIDModel
    model = LIDModel(); model.load()
    labels = model.predict(["yaar", "I", "am", "coming"])
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hinglish.features.phonetic_matcher import get_phonetic_hash

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_FILE = os.path.join(DATA_DIR, "lid_model.json")

SMOOTHING  = 1e-8
CONTEXT_W  = 2          # window half-width for context smoothing
CONTEXT_WT = 0.15       # weight given to neighbour votes


class LIDModel:
    """Naive Bayes LID model with character-ngram + phonetic fallback."""

    def __init__(self):
        self.word_probs:     dict[str, dict[str, float]] = {}   # word  → {label: logP}
        self.phonetic_probs: dict[str, dict[str, float]] = {}   # phash → {label: logP}
        self.class_probs:    dict[str, float] = {}               # label → logP(label)
        self.labels:         list[str] = []
        self.seen_words:     set[str]  = set()

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, data_path: str):
        word_counts:     defaultdict = defaultdict(lambda: defaultdict(int))
        phonetic_counts: defaultdict = defaultdict(lambda: defaultdict(int))
        class_counts:    defaultdict = defaultdict(int)
        total_tokens = 0

        with open(data_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tok = row["token"].strip().lower()
                lbl = row["label"].strip()
                if not tok or not lbl:
                    continue
                word_counts[tok][lbl] += 1
                phonetic_counts[get_phonetic_hash(tok)][lbl] += 1
                class_counts[lbl] += 1
                total_tokens += 1

        self.labels = sorted(class_counts.keys())
        vocab_size   = len(word_counts)
        phash_size   = len(phonetic_counts)

        # class priors (log)
        self.class_probs = {
            lbl: math.log(cnt / total_tokens)
            for lbl, cnt in class_counts.items()
        }

        # word → label log-probabilities  (add-1 / Laplace smoothing)
        self.word_probs = {}
        for word, lbl_counts in word_counts.items():
            total = sum(lbl_counts.values())
            self.word_probs[word] = {
                lbl: math.log((lbl_counts.get(lbl, 0) + 1) / (total + len(self.labels)))
                for lbl in self.labels
            }
        self.seen_words = set(self.word_probs)

        # phonetic-hash → label log-probabilities
        self.phonetic_probs = {}
        for ph, lbl_counts in phonetic_counts.items():
            total = sum(lbl_counts.values())
            self.phonetic_probs[ph] = {
                lbl: math.log((lbl_counts.get(lbl, 0) + 1) / (total + len(self.labels)))
                for lbl in self.labels
            }

        print(f"LID model trained:  {vocab_size} words  |  {phash_size} phonetic hashes  |  labels={self.labels}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "labels":         self.labels,
            "class_probs":    self.class_probs,
            "word_probs":     self.word_probs,
            "phonetic_probs": self.phonetic_probs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"LID model saved → {path}")

    def load(self, path: str = MODEL_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"LID model not found at {path}. Run with --train first.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.labels         = data["labels"]
        self.class_probs    = data["class_probs"]
        self.word_probs     = data["word_probs"]
        self.phonetic_probs = data["phonetic_probs"]
        self.seen_words     = set(self.word_probs)

    # ── inference ─────────────────────────────────────────────────────────────

    def _score_token(self, tok: str) -> dict[str, float]:
        """Return log-probability scores {label: score} for a single token."""
        low = tok.lower()
        if low in self.word_probs:
            scores = dict(self.word_probs[low])
        else:
            ph = get_phonetic_hash(low)
            if ph in self.phonetic_probs:
                scores = dict(self.phonetic_probs[ph])
            else:
                scores = {lbl: self.class_probs[lbl] for lbl in self.labels}

        # add class prior
        for lbl in self.labels:
            scores[lbl] = scores.get(lbl, math.log(SMOOTHING)) + self.class_probs.get(lbl, math.log(SMOOTHING))
        return scores

    def predict(self, tokens: list[str]) -> list[str]:
        """Predict a LID label per token with sliding-window context smoothing."""
        if not tokens:
            return []

        PUNC = set(".,!?;:()[]{}—–-\"'")

        # --- pass 1: per-token scores ---
        raw_scores = []
        for tok in tokens:
            if not tok.replace("'", "").isalpha():
                raw_scores.append({lbl: (1.0 if lbl == "UNI" else 0.0) for lbl in self.labels})
            else:
                raw_scores.append(self._score_token(tok))

        # --- pass 2: context smoothing ---
        labels_out = []
        for i, scores in enumerate(raw_scores):
            # aggregate neighbour votes
            neighbour_votes: dict[str, float] = {lbl: 0.0 for lbl in self.labels}
            count = 0
            for j in range(max(0, i - CONTEXT_W), min(len(raw_scores), i + CONTEXT_W + 1)):
                if j == i:
                    continue
                best_nb = max(raw_scores[j], key=raw_scores[j].get)
                neighbour_votes[best_nb] += 1.0
                count += 1

            # blend: own score (linear) + context
            own_best = max(scores, key=scores.get)
            if count == 0:
                labels_out.append(own_best)
                continue

            blended: dict[str, float] = {}
            own_max = max(scores.values())
            for lbl in self.labels:
                own_norm   = math.exp(scores[lbl] - own_max)      # softmax-like
                ctx_weight = neighbour_votes[lbl] / count
                blended[lbl] = (1 - CONTEXT_WT) * own_norm + CONTEXT_WT * ctx_weight

            labels_out.append(max(blended, key=blended.get))

        return labels_out


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, random

    parser = argparse.ArgumentParser(description="Train or evaluate the LID model.")
    parser.add_argument("--train",    action="store_true", help="Train on 80% of train_lid.csv")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on bis_test_lid.csv")
    parser.add_argument("--data",     default="data/raw/train_lid.csv")
    parser.add_argument("--test",     default="data/raw/bis_test_lid.csv")
    args = parser.parse_args()

    model = LIDModel()

    if args.train:
        model.train(args.data)
        model.save()

    if args.evaluate:
        model.load()
        correct = total = 0
        with open(args.test, encoding="utf-8") as f:
            sentences, cur = [], []
            for row in csv.DictReader(f):
                tok, lbl = row["token"].strip(), row["label"].strip()
                if not tok or not lbl:
                    if cur: sentences.append(cur); cur = []
                else: cur.append((tok, lbl))
            if cur: sentences.append(cur)

        for sent in sentences:
            tokens = [t for t, _ in sent]
            truth  = [l for _, l in sent]
            preds  = model.predict(tokens)
            for t, p in zip(truth, preds):
                if t == p: correct += 1
                total += 1
        print(f"\nLID Accuracy: {correct/total:.2%}  ({correct}/{total})")
