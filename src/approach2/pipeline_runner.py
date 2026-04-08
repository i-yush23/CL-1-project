
import os
import re
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from approach2.statistical_models.code_switch_labeler import CodeswitchSegmenter
from approach2.statistical_models.hmm_pos_tagger import HMMPosTagger

def run_pipeline(text):
    segmenter = CodeswitchSegmenter()
    tagger = HMMPosTagger()
    tagger.load() # Uses pre-trained weights

    # Tokenize (handling punctuation)
    # The regex finds words or specific punctuation
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    
    # Process
    lid_labels = segmenter.segment(tokens)
    pos_results = tagger.viterbi_decode(tokens)
    
    print(f"\n{'Token':15} | {'LID':5} | {'POS':5}")
    print("-" * 35)
    for i in range(len(tokens)):
        token = tokens[i]
        lid = lid_labels[i]
        pos = pos_results[i][1]
        print(f"{token:15} | {lid:5} | {pos:5}")

if __name__ == "__main__":
    test_text = "Yaar, kal raat ko maine ek nayi web series start ki, but it was so boring ki mujhe neend aa gayi. Aaj boss ne mujhe urgent report submit karne ko kaha hai, aur main abhi tak uske baare mein soch raha hoon. Pata nahi ye project kab complete hoga. Anyway, coffee break ka time ho gaya hai, chalte hain!"
    run_pipeline(test_text)
