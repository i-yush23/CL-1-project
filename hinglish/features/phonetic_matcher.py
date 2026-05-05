
import re
import json
from collections import defaultdict
from tqdm import tqdm
def get_phonetic_hash(word: str) -> str:
    """
    Converts a Hinglish word into its phonetic skeleton to match variants.
    e.g., 'bhaaaaai', 'bhai', 'bhya' -> 'BH'
    """
    if not word or not word.isalpha():
        # Return as-is if it's empty, punctuation, or numbers
        return word.lower()

    # 1. Lowercase
    w = word.lower()

    # 2. Compress all repeated characters (aa -> a, bb -> b)
    # The regex (i.e., "(.)\1+") means "find any character and any identical characters immediately following it"
    w = re.sub(r'(.)\1+', r'\1', w)

    # 3. Phonetic Grouping (Handle multi-character sounds FIRST)
    w = w.replace("sh", "S")
    w = w.replace("ph", "F")
    w = w.replace("ch", "C")
    
    # Then handle single-character sounds
    phonetic_map = {
        'k': 'K', 'c': 'K', 'q': 'K',
        'v': 'W', 'w': 'W',
        's': 'S', 'z': 'S',
        'f': 'F'
    }
    # Translate matching characters to their grouped uppercase version
    # Letters not in the map stay lowercase for now
    for char, replacement in phonetic_map.items():
        w = w.replace(char, replacement)

    # 4. Vowel Dropping (Keep the first letter, drop the rest if they are vowels)
    # If the word was reduced to 1 letter after compression, just return it
    if len(w) <= 1:
        return w.upper()
    
    first_char = w[0]
    remaining = w[1:]
    
    # Remove a, e, i, o, u, y from the remainder of the word
    remaining_no_vowels = re.sub(r'[aeiouy]', '', remaining)
    
    # Combine and return fully uppercased for the final hash
    final_hash = (first_char + remaining_no_vowels).upper()
    
    return final_hash
import os

# Load transliteration variants
fallback_path = os.path.join(os.path.dirname(__file__), "..", "data", "transliteration_variants.json")
try:
    with open(fallback_path, "r") as f:
        fallback_dict = json.load(f)
except FileNotFoundError:
    fallback_dict = {}

def normalize_word(word: str) -> str:
    """
    Applies the fast O(1) fallback dictionary check for known transliterations.
    If the word is not in the dictionary, returns its phonetic hash.
    """
    word_lower = word.lower()
    if word_lower in fallback_dict:
        return fallback_dict[word_lower]
    return get_phonetic_hash(word_lower)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    phonetic_bigrams = defaultdict(lambda: defaultdict(float))
    with open(os.path.join(current_dir, "bigram_probs.json"), "r") as file:
        bigram_probs = json.load(file)
        for prev_word, word_dict in tqdm(bigram_probs.items(), total=len(bigram_probs), desc="Processing Bigrams"):
            for next_word, prob in word_dict.items():
                prev_hash = normalize_word(prev_word)
                next_hash = normalize_word(next_word)
                phonetic_bigrams[prev_hash][next_hash] += prob

    for prev_hash, next_hash_dict in phonetic_bigrams.items():
        row_sum = sum(next_hash_dict.values())
        if row_sum > 0:
            for next_hash in next_hash_dict:
                phonetic_bigrams[prev_hash][next_hash] /= row_sum

    phonetic_trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with open(os.path.join(current_dir, "trigram_probs.json"), "r") as file:
        trigram_probs = json.load(file)

        for context_key, next_word_dict in tqdm(trigram_probs.items(), total=len(trigram_probs), desc="Processing Trigrams"):
            prev_prev_word, prev_word = context_key.split("|")
            prev_prev_hash = normalize_word(prev_prev_word)
            prev_hash = normalize_word(prev_word)
            for next_word, prob in next_word_dict.items():
                next_hash = normalize_word(next_word)
                phonetic_trigrams[prev_prev_hash][prev_hash][next_hash] += prob

    for prev_prev_hash, prev_hash_dict in phonetic_trigrams.items():
        for prev_hash, next_hash_dict in prev_hash_dict.items():
            row_sum = sum(next_hash_dict.values())
            if row_sum > 0:
                for next_hash in next_hash_dict:
                    phonetic_trigrams[prev_prev_hash][prev_hash][next_hash] /= row_sum


    with open(os.path.join(current_dir, "phonetic_bigram_probs.json"), "w") as file:
        json.dump(phonetic_bigrams, file, indent=2)
    with open(os.path.join(current_dir, "phonetic_trigram_probs.json"), "w") as file:
        json.dump(phonetic_trigrams, file, indent=2)
