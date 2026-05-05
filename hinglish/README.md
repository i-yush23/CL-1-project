# Implementation Plan: Purely Statistical Pipeline (Approach 2)

This document outlines the architecture and execution plan for building a purely statistical and rule-based pipeline for Hinglish text processing, replacing the CRF/ML models.

## 1. Project Restructuring
The previous CRF-based machine learning pipeline has been isolated into `src/approach1/`.
The new components will be built in `hinglish/`.

### Directory Structure
```
hinglish/
├── data/
│   ├── bigram_probs.json
│   ├── trigram_probs.json
│   ├── transliteration_variants.json
│   └── emoji_lexicon.json
├── features/
│   ├── phonetic_matcher.py
│   ├── ngram_analyzer.py
│   └── emoji_analyzer.py
├── statistical_models/
│   ├── hmm_pos_tagger.py
│   └── lid_model.py
├── tools/
│   └── evaluate.py
├── demo.py
└── pipeline_runner.py
```

---

## 2. Component Breakdown

### A. Frequency-Based Corrections (Bigram/Trigram Analysis)
**Concept:** Use Markov chains and n-gram probabilities instead of feature-engineered CRFs.
- **Implementation:**
  - Build bigram and trigram frequency tables from `hinglish_crf_train_data.jsonl`.
  - To correct spelling or infer a tag, calculate the maximum likelihood path using the Viterbi algorithm.
  - E.g., if we see "kya haal", the bigram probability of `P("hai" | "kya", "haal")` can be used to statistically correct "he" or "h" into "hai".

### B. Phonetic Matching for Romanized Text
**Concept:** Map phonetically similar transliterations to a common canonical form.
- **Implementation:**
  - Create a custom **Metaphone/Soundex algorithm for Hinglish**.
  - Drop all vowels (except at the start), compress repeated consonants, and group similar-sounding consonants (e.g., `k`/`c`/`q` $\rightarrow$ `K`, `v`/`w` $\rightarrow$ `W`).
  - Calculate Levenshtein edit distance with dynamically adjusted weights (e.g., substituting 'a' for 'e' costs 0.5, but substituting 'k' for 'p' costs 2.0).

### C. Common Transliteration Variants Fallback
**Concept:** Keep a static dictionary for the most common words that bypasses complex calculations.
- **Implementation:**
  - Maintain a JSON mapping of frequent variants to a root form.
  - Examples:
    - `"kyu", "kyun", "kyon", "kio" -> "kyun"`
    - `"nh", "nai", "nhi", "nahin" -> "nahi"`
  - This dictionary is always checked *before* falling back to the phonetic matcher or n-gram models.

### D. Label Code-Switched Segments for Targeted Processing
**Concept:** Instead of word-by-word LID classification via CRF, label continuous blocks/segments of text statistically.
- **Implementation:**
  - Evaluate the "Englishness" vs "Hindiness" of a rolling window of 3 words.
  - Transition probabilities: the penalty for switching languages is statistically modeled. If a block is identified as purely Hindi transliteration, apply Hindi POStagging rules.
  - Output explicit segment spans: `[0:4] -> HI`, `[4:7] -> EN`.

### E. Emoji Sentiment Segregation
**Concept:** Use fixed rules to classify the sentiment of recognized emojis to aid downstream tasks.
- **Implementation:**
  - Build `emoji_analyzer.py` mapping Unicode emojis to `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.
  - Extract emojis during tokenization and inject a sentiment placeholder (e.g., `__EMOJI_POS__`) to modify the text's statistical probabilities or tag sequences.

---

## 3. Development Phases

- **Phase 1: Foundations**
  - Extract n-gram frequencies from the raw dataset.
  - Build the fallback dictionary (`transliteration_variants.json`).
- **Phase 2: Phonetic & Token Features**
  - Implement the Hinglish phonetic hashing logic.
  - Add the emoji classification dictionaries.
- **Phase 3: Core Statistical Algorithms**
  - Implement the Code-Switch Segmenter.
  - Implement an HMM-based (Hidden Markov Model) POS tagger using the frequencies derived in Phase 1.
- **Phase 4: Pipeline Assembly**
  - Wire the steps together in `pipeline_runner.py` to accept raw text and output POS tags + Segment labels entirely through statistics.

---

## 4. Usage and Testing

To test the model and evaluate its performance, you can use the following scripts. All commands should be run from the project root.

### A. Run Interactive Demo
The demo provides a quick way to see the pipeline in action with a sample Hinglish sentence.
```bash
python hinglish/demo.py
```

### B. Run Pipeline Runner
You can also run the pipeline runner directly, which contains a more complex test string.
```bash
python hinglish/pipeline_runner.py
```

### C. Evaluate Model Accuracy
To generate a full accuracy report for both Language Identification (LID) and POS tagging on the test sets:
```bash
python hinglish/tools/evaluate.py
```
This script will output:
- Overall accuracy for LID and POS tagging.
- Precision, Recall, and F1-score for each label.
- Top confusions (errors) for both sub-systems.
- OOV (Out-of-Vocabulary) performance breakdown for the POS tagger.

### D. Data Preparation (Optional)
If you need to re-generate the test sets from raw data:
```bash
python hinglish/tools/make_hinglish_lid.py
python hinglish/tools/make_bis_test.py
```
