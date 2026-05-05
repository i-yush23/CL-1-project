# CL-1 Project — Code-Mixed NLP Pipelines

Purely statistical NLP pipelines for two code-mixed language pairs:

| Pipeline | Languages | LID Accuracy | POS Accuracy |
|----------|-----------|:------------:|:------------:|
| **Hinglish** | Hindi ↔ English (romanised) | **97.15%** | **71.18%** |
| **Tenglish** | Telugu ↔ English (romanised) | — | — |

> **New modules added:** Code-Mixing Index (CMI) · Emoji Analyzer · Matrix Language Classifier

Both pipelines are **Approach 2** implementations: no neural networks, no pre-trained embeddings — only Naive Bayes / HMM statistical models with phonetic hashing.

---

## Repository Layout

```
CL-1-project/
├── hinglish/                   ← Hindi-English pipeline
│   ├── data/                   ← Trained model JSON files
│   ├── features/
│   │   ├── phonetic_matcher.py ← Hinglish phonetic hasher
│   │   └── emoji_analyzer.py   ← Emoji → sentiment label
│   ├── metrics/
│   │   └── cmi_calculator.py   ← Code-Mixing Index (CMI)
│   ├── statistical_models/
│   │   ├── lid_model.py        ← Naive Bayes LID (EN/HI/UNI)
│   │   ├── hmm_pos_tagger.py   ← HMM Viterbi POS tagger
│   │   └── matrix_classifier.py← MLF-based Matrix Language Classifier
│   ├── features/
│   │   ├── phonetic_matcher.py ← Hinglish phonetic hasher
│   │   └── emoji_analyzer.py   ← Emoji → sentiment placeholder
│   ├── tools/
│   │   ├── evaluate.py         ← Full accuracy report
│   │   ├── make_hinglish_lid.py← Build LID train/test CSVs
│   │   └── make_bis_test.py    ← Build BIS-tagged test sets
│   ├── pipeline_runner.py      ← End-to-end pipeline (CLI + REPL)
│   ├── demo.py                 ← Quick interactive demo
│   └── README.md               ← Hinglish implementation plan
│
├── tenglish/                   ← Telugu-English pipeline
│   ├── data/raw/               ← Training & test CSVs
│   └── src/approach2/
│       ├── data/               ← Trained model JSON files
│       ├── features/
│       │   └── phonetic_matcher.py  ← Telugu-aware phonetic hasher
│       ├── statistical_models/
│       │   ├── code_switch_labeler.py ← LID (TE/EN/UNI)
│       │   └── hmm_pos_tagger.py      ← HMM Viterbi POS tagger
│       ├── tools/
│       │   ├── verify_hmm_accuracy.py ← Evaluate LID + POS
│       │   └── regenerate_indicators.py
│       └── pipeline_runner.py  ← End-to-end pipeline
│
├── data/raw/                   ← Shared BIS/Hinglish corpus splits
│   ├── train_pos.csv           ← HMM POS training data
│   ├── train_lid.csv           ← LID training data
│   ├── bis_test_pos.csv        ← POS held-out test set
│   └── hinglish_test_lid.csv   ← LID held-out test set (271k tokens)
│
├── hinglish_crf_train_data.jsonl ← Raw Hinglish annotated corpus
├── hinglish_report.pdf           ← Detailed evaluation report
└── requirements.txt
```

---

## Setup

```bash
# Install shared dependencies (from project root)
pip install tqdm

# Hinglish has no extra deps beyond the above
# Tenglish needs:
pip install -r tenglish/requirements.txt
```

> **Python 3.11+** required.

---

## Hinglish Pipeline (Hindi ↔ English)

All commands run from the **project root**.

### Run the demo
```bash
python hinglish/demo.py
```

### Run with a sentence argument
```bash
cd CL-1-project
python hinglish/pipeline_runner.py "Yaar I cannot believe this hogaya 😂, itna boring tha"
```

### Run interactively (REPL mode)
```bash
cd CL-1-project
python hinglish/pipeline_runner.py
```
Type any Hinglish sentence at the `>` prompt, then `quit` to exit.

**Output:**
```
───────────────────────────────────────────────────────
  INPUT : Yaar I cannot believe this hogaya 😂, itna boring tha
───────────────────────────────────────────────────────

  Token                  LID     POS
  ─────────────────────────────────────────────
  Yaar                   HI      N
  I                      EN      PRP
  cannot                 EN      V
  believe                EN      V
  this                   EN      PRP
  hogaya                 HI      N
  😂                      EMOJI   EMOJI
  ,                      UNI     PUNC
  itna                   HI      PRT
  boring                 EN      J
  tha                    HI      V

  ─────────────────────────────────────────────
  CMI Score        : 40.0 / 100
  HI tokens        : 4   EN tokens: 6   UNI tokens: 1

  Matrix Language  : EN
  Embedded Language: HI
  HI score: 6.0   EN score: 10.0

  Emojis detected  : 😂
  Grammar anchors  : I, cannot, believe, this, itna, tha
───────────────────────────────────────────────────────
```

### Import in your own script
```python
import sys
sys.path.insert(0, "CL-1-project")
from hinglish.pipeline_runner import run_pipeline

run_pipeline("Kal main office nahi aaunga, but I'll join the meeting online.")
```

### Evaluate (LID + POS accuracy report)
```bash
python hinglish/tools/evaluate.py
```
Reports overall accuracy, per-class Precision / Recall / F1, and top confusions for both LID and POS.

### Retrain models (optional)
```bash
# Rebuild LID train/test splits from the raw JSONL corpus
python hinglish/tools/make_hinglish_lid.py
python hinglish/tools/make_bis_test.py

# Retrain the Naive Bayes LID model
python hinglish/statistical_models/lid_model.py --train

# Retrain the HMM POS tagger
python hinglish/statistical_models/hmm_pos_tagger.py
```

---

## Tenglish Pipeline (Telugu ↔ English)

Commands run from the **`tenglish/` directory**.

```bash
cd tenglish
```

### Run with a sentence argument
```bash
python src/approach2/pipeline_runner.py "Nuvvu ikkade undi, but I thought you were going."
```

### Run interactively (built-in example sentence)
```bash
python src/approach2/pipeline_runner.py
```

**Output:**
```
Token                | LID    | POS
------------------------------------------
Nuvvu                | TE     | PRON
ikkade               | TE     | ADV
undi                 | TE     | V
,                    | UNI    | PUNC
but                  | EN     | CONJ
I                    | EN     | PRON
thought              | EN     | V
...
```

### Evaluate
```bash
python src/approach2/tools/verify_hmm_accuracy.py --pos data/raw/test_pos.csv
python src/approach2/tools/verify_hmm_accuracy.py --lid data/raw/test_lid.csv
```

### Regenerate indicator wordlists
```bash
python src/approach2/tools/regenerate_indicators.py
```

---

## Pipeline Comparison

| Feature | Hinglish | Tenglish |
|---------|----------|----------|
| **LID labels** | `EN`, `HI`, `UNI` | `EN`, `TE`, `UNI`, `EMOJI` |
| **POS tagset** | BIS/IIIT (`N`, `V`, `PSP`, `G_N`, …) | Universal (`N`, `V`, `ADP`, `CONJ`, …) |
| **LID model** | Naive Bayes + phonetic hash | Rule-based indicator sets + phonetic hash |
| **POS model** | HMM Viterbi | HMM Viterbi |
| **Phonetic hashing** | Hinglish-tuned (sh/ph/ch clusters) | Telugu-tuned |
| **Emoji handling** | Detected → `__EMOJI_SENTIMENT__` placeholder | Gets `EMOJI` LID label |
| **CMI** | Computed post-LID via `CmiCalculator` | — |
| **Matrix Language** | MLF-based `MatrixClassifier` | — |
| **Slang expansion** | Not applied | `slang_map.json` applied pre-LID |
| **Context smoother** | Sliding window ±2 (weight 0.15) | — |
| **OOV fallback** | Phonetic hash → class prior | Phonetic hash → indicator lookup |

---

## How the Models Work

### Language Identification (LID)

**Hinglish** uses a trained **Naive Bayes** model over a 66,828-word vocabulary:
1. Exact word lookup in vocabulary → log P(label | word)
2. Phonetic hash lookup (22,803 buckets) — handles spelling variants
3. Class prior fallback (HI ≈ 71%, EN ≈ 28%, UNI ≈ 1%)
4. Sliding-window context smoother blends neighbour votes (±2 tokens, 15% weight)

**Tenglish** uses a **rule-based indicator** approach:
1. Non-alpha → `UNI`
2. Transliteration variants dictionary
3. Telugu indicator word list / English indicator word list
4. Phonetic hash → indicator lookup for unseen spellings
5. Default: `TE`

### POS Tagging (both pipelines)

Both use a **first-order Hidden Markov Model** with Viterbi decoding:
- Emission: `P(word | tag)` — exact word lookup, else phonetic hash, else tag-frequency prior
- Transition: `P(tag_t | tag_{t-1})`
- Hard constraint: alphabetic tokens never get `SYM`/`PUNC` tags

---

## Results

See [`hinglish_report.pdf`](hinglish_report.pdf) for the full evaluation report on the Hinglish pipeline, including per-class metrics, confusion matrices, and architectural analysis.

### Hinglish — LID

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| EN | 98.72% | 90.73% | 94.56% | 74,135 |
| HI | 96.55% | 99.55% | 98.03% | 193,475 |
| UNI | 100.00% | 100.00% | 100.00% | 3,941 |
| **Overall** | | | **97.15%** | 271,551 |

### Hinglish — POS

| Metric | Accuracy |
|--------|----------|
| Overall | 71.18% (2,870 / 4,032) |
| In-Vocabulary | 79.62% (2,653 / 3,332) |
| Out-of-Vocabulary | 31.00% (217 / 700) |
