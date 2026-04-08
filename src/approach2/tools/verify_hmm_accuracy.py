
import csv
import os
import sys
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from approach2.statistical_models.hmm_pos_tagger import HMMPosTagger

def verify_hmm_accuracy(data_path="data/raw/train_pos.csv"):
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    # Load pre-trained models
    tagger = HMMPosTagger()
    tagger.load() 
    if not tagger.all_tags:
        print("Error: Could not load HMM tags. Is it trained?")
        return

    sentences = []
    current_sentence = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['token'].strip()
            tag = row['label'].strip()
            
            if not word or not tag:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            current_sentence.append((word, tag))
        if current_sentence:
            sentences.append(current_sentence)

    correct = 0
    total = 0
    
    for sent in tqdm(sentences, desc="Verifying Accuracy"):
        tokens = [pair[0] for pair in sent]
        true_tags = [pair[1] for pair in sent]
        
        predictions = tagger.viterbi_decode(tokens)
        predicted_tags = [pair[1] for pair in predictions]
        
        for p, t in zip(predicted_tags, true_tags):
            if p == t:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nHMM POS Tagger Training Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}, Total Tokens: {total}")

if __name__ == "__main__":
    verify_hmm_accuracy()
