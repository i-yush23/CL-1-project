
import csv
import random
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from approach2.statistical_models.hmm_pos_tagger import HMMPosTagger

def random_test(data_path="data/raw/train_pos.csv", num_samples=5):
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    # Load all sentences
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

    # Pick random samples
    if num_samples > len(sentences):
        num_samples = len(sentences)
    samples = random.sample(sentences, num_samples)

    tagger = HMMPosTagger()
    tagger.load()

    overall_correct = 0
    overall_total = 0

    print(f"\n--- Testing on {num_samples} Random Samples ---")
    
    for i, sent in enumerate(samples):
        tokens = [pair[0] for pair in sent]
        true_tags = [pair[1] for pair in sent]
        
        # Hard Fix for Punctuation in ground truth comparison
        # (Since the user wants PUNC but the truth might have SYM/N/V)
        PUNCTUATIONS = set([".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "--", "..."])
        sanitized_true = []
        for tok, tag in sent:
            if tok in PUNCTUATIONS: sanitized_true.append("PUNC")
            else: sanitized_true.append(tag)

        predictions = tagger.viterbi_decode(tokens)
        predicted_tags = [pair[1] for pair in predictions]

        print(f"\nSample {i+1}: {' '.join(tokens)}")
        print(f"{'Token':15} | {'True':8} | {'Pred':8} | {'Status'}")
        print("-" * 50)
        
        sent_correct = 0
        for tok, t, p in zip(tokens, sanitized_true, predicted_tags):
            status = "✅" if t == p else "❌"
            if t == p:
                sent_correct += 1
                overall_correct += 1
            overall_total += 1
            print(f"{tok:15} | {t:8} | {p:8} | {status}")
        
        sent_acc = (sent_correct / len(tokens)) * 100
        print(f"Sentence Accuracy: {sent_acc:.2f}%")

    total_acc = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print(f"\n{'='*50}")
    print(f"OVERALL ACCURACY FOR SAMPLES: {total_acc:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    random_test(num_samples=5)
