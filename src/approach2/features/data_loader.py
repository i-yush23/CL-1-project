from huggingface_hub import TextGenerationInputGrammarType
import json
from collections import defaultdict
from tqdm import tqdm

unigram_counts = defaultdict(int)

bigram_counts = defaultdict(lambda:defaultdict(int))

trigram_counts = defaultdict(lambda:defaultdict(int))
total_lines = sum(1 for _ in open("../../../hinglish_crf_train_data.jsonl", "r"))
with open("../../../hinglish_crf_train_data.jsonl", "r") as file:
    for line in tqdm(file, total = total_lines, desc="Processing N-grams"):
        data = json.loads(line)
        tokens = data["tokens"]

        tokens = [t.lower() for t in tokens]

        for i in range(len(tokens)):
            current_word = tokens[i]

            unigram_counts[current_word]+=1

            if i>0:
                prev_word = tokens[i-1]
                bigram_counts[prev_word][current_word]+=1
            if i>1:
                prev_prev_word = tokens[i-2]
                prev_word = tokens[i-1]

                context_key = f"{prev_prev_word}|{prev_word}"
                trigram_counts[context_key][current_word]+=1
        

bigram_probs =  defaultdict(dict)
trigram_probs = defaultdict(dict)

for prev_word, next_word_dict in bigram_counts.items():
    total_prev_word_occurances = unigram_counts[prev_word]

    for next_word, count in next_word_dict.items():
        probability = count / total_prev_word_occurances
        bigram_probs[prev_word][next_word] = round(probability,5)

for context_key, next_word_object in trigram_counts.items():

    prev_prev_word, prev_word = context_key.split("|")
    total_context_occurances =  bigram_counts[prev_prev_word][prev_word]

    for next_word, count in next_word_object.items():
        probability = count / total_context_occurances
        trigram_probs[context_key][next_word] = round(probability,5)
    
with open("bigram_probs.json", "w") as file:
    json.dump(bigram_probs, file, indent=2)
with open("trigram_probs.json", "w") as file:
    json.dump(trigram_probs, file, indent=2)

