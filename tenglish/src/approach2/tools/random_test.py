"""
random_test.py
Quick sanity-check tool — runs the full Tenglish pipeline on a set of
hard-coded sentences and prints results for manual inspection.

Usage:
    python random_test.py
"""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from approach2.statistical_models.code_switch_labeler import CodeswitchSegmenter
from approach2.statistical_models.hmm_pos_tagger      import HMMPosTagger


TEST_SENTENCES = [
    # Everyday code-mixed chat
    "Nuvvu ikkade undi, but I thought you went to college.",
    "Naaku cheppavu kaadu em? That's not fair bro.",
    "Ippudu chaala busy ga unnanu, will call you later.",
    # Heavy Telugu
    "Vaadu office ki vellaadu, ayyo naaku teliyadu ikkade undi ani.",
    "Meeru ela unnaru? Baagundi aa?",
    # Heavy English
    "I have no idea what is happening here to be honest.",
    "Please send me the file by tomorrow morning okay?",
    # Interjections & particles
    "Ayyo arre, ippudu em cheyyali?",
    "Abba konchem wait cheyyi, I am coming.",
    # Numbers & punctuation stress-test
    "3 rojulu ayindi nuvvu raakunda, what happened?",
    "Meeting 9am ki undi, ledu aa?",
]


def main():
    segmenter = CodeswitchSegmenter()
    tagger    = HMMPosTagger()
    tagger.load()

    for sentence in TEST_SENTENCES:
        tokens     = re.findall(r"[\w']+|[.,!?;:—…]", sentence)
        lid_labels = segmenter.segment(tokens)
        pos_tags   = tagger.viterbi_decode(tokens)

        print(f"\nInput : {sentence}")
        print(f"{'Token':<20} | {'LID':<5} | {'POS'}")
        print("-" * 38)
        for tok, lid, (_, pos) in zip(tokens, lid_labels, pos_tags):
            print(f"{tok:<20} | {lid:<5} | {pos}")


if __name__ == "__main__":
    main()
