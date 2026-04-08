
import sys
import os
import re

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from approach2.statistical_models.code_switch_labeler import CodeswitchSegmenter

def test_segmenter():
    segmenter = CodeswitchSegmenter()
    test_sentences = [
        "bohut achay ayay",
        "listening to Ishq Wala Love",
        "Bhai koi pehchan wala song compile kr le",
        "It is indeed when its possible"
    ]
    
    for sentence in test_sentences:
        tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
        labels = segmenter.segment(tokens)
        print(f"Sentence: {sentence}")
        print(f"Tokens:   {tokens}")
        print(f"Labels:   {labels}")
        print("-" * 30)

if __name__ == "__main__":
    test_segmenter()
