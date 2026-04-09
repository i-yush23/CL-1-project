# Tenglish NLP Pipeline — Telugu-English Code-Mixed Text

Approach 2 — HMM-based Language Identification + POS Tagging
(Adapted from the Hinglish CL-1 project)

---

## Project Structure

```
tenglish-project/
├── data/
│   └── raw/
│       ├── train_pos.csv     ← your annotated training data (token, label)
│       └── test_pos.csv      ← your annotated test data
├── src/
│   └── approach2/
│       ├── data/             ← auto-generated model files (JSON)
│       │   ├── hmm_emissions.json
│       │   ├── hmm_phonetic_emissions.json
│       │   ├── hmm_transitions.json
│       │   ├── hmm_unigrams.json
│       │   ├── te_indicators.json
│       │   ├── en_indicators.json
│       │   └── transliteration_variants.json
│       ├── features/
│       │   └── phonetic_matcher.py   ← Telugu-aware phonetic hashing
│       ├── statistical_models/
│       │   ├── code_switch_labeler.py  ← LID (TE / EN / UNI)
│       │   └── hmm_pos_tagger.py       ← Viterbi POS tagger
│       ├── tools/
│       │   ├── regenerate_indicators.py
│       │   ├── random_test.py
│       │   └── verify_hmm_accuracy.py
│       └── pipeline_runner.py
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate indicator wordlists
```bash
python src/approach2/tools/regenerate_indicators.py
```

### 3. Train the HMM POS tagger
Place your annotated CSV at `data/raw/train_pos.csv` with columns `token,label`.
Blank rows mark sentence boundaries.

```bash
cd <project-root>
python src/approach2/statistical_models/hmm_pos_tagger.py
```

### 4. Run the pipeline
```bash
python src/approach2/pipeline_runner.py "Nuvvu ikkade undi, but I thought you were going to the market."

# or interactively:
python src/approach2/pipeline_runner.py
```

### 5. Evaluate
```bash
python src/approach2/tools/verify_hmm_accuracy.py --pos data/raw/test_pos.csv
python src/approach2/tools/verify_hmm_accuracy.py --lid data/raw/test_lid.csv
```

---

## Pipeline Overview

```
Raw Tenglish text
       │
       ▼
   Tokenize          regex: [\w']+ | punctuation
       │
       ▼
CodeswitchSegmenter  → LID label per token (TE / EN / UNI)
  1. Hard-rule: non-alpha → UNI
  2. Transliteration fallback dict lookup
  3. Indicator set membership (te_indicators, en_indicators)
  4. Phonetic-hash fallback for unseen spellings
  5. Context smoothing for unknowns
  6. Default: TE
       │
       ▼
 HMMPosTagger        → POS tag per token (Viterbi decoding)
  Emission:  P(word | tag)  +  phonetic-hash fallback
  Transition: P(tag_t | tag_{t-1})
  Smoothing: 1e-10
```

---

## POS Tagset

| Tag   | Description                              | Example (Tenglish)       |
|-------|------------------------------------------|--------------------------|
| N     | Noun                                     | illu, office, time       |
| V     | Verb                                     | chestunnav, went, undi   |
| ADJ   | Adjective                                | chaala, busy, good       |
| ADV   | Adverb                                   | ippudu, later, already   |
| PRON  | Pronoun                                  | nuvvu, I, vaadu, meeru   |
| DET   | Determiner / Article                     | the, a, oka              |
| ADP   | Adposition (Telugu postpositions)        | lo, ki, tho, nunchi      |
| CONJ  | Conjunction                              | kani, but, and, ledu     |
| PART  | Particle                                 | ani, ante, ayithe        |
| PROPN | Proper Noun                              | Hyderabad, Ravi, Google  |
| NUM   | Number                                   | 3, rendu, ten            |
| PUNC  | Punctuation                              | . , ! ?                  |
| INTJ  | Interjection                             | ayyo, arre, abba         |
| X     | Foreign / unclassified                   | lol, omg                 |

---

## Where to Get Training Data

### Annotated Telugu-English datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **ICON 2016 Shared Task** | Telugu-English code-mixed POS + LID | Search "ICON 2016 Telugu English codemixed" |
| **IIT Bombay Code-Mixed** | Multi-lingual incl. Telugu-English | https://www.cfilt.iitb.ac.in |
| **Dravidian-CodeMix (FIRE)** | Telugu-English sentiment & NER | https://dravidian-codemix.github.io |
| **CALCS workshop datasets** | Code-switching shared tasks | https://code-switching.github.io |
| **LINC corpus** | Telugu social media text | Contact authors via ACL Anthology |

### Build your own training CSV
Annotate Tenglish tweets/messages in this format:
```
token,label
Nuvvu,PRON
ikkade,ADV
undi,V
,
but,CONJ
I,PRON
...
```
Recommended tool: **brat** (https://brat.nlplab.org) or a simple spreadsheet.

### Raw unannotated Telugu-English text sources
- Twitter/X: search Telugu hashtags (#Telugu, #Tollywood)
- YouTube comments on Telugu movie trailers
- WhatsApp public group exports
- Reddit: r/telugu

Use scraped text to build/expand `transliteration_variants.json` and the indicator lists.
