"""
generate_report.py - Generates a detailed PDF report for the Hinglish NLP pipeline.
Run from project root: python tmp/generate_report.py
"""

import os, sys, math
sys.path.insert(0, os.path.abspath("."))

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ??????????????????????????????????????????????????
# COLOUR PALETTE
# ??????????????????????????????????????????????????
C_NAVY    = (15,  30,  72)
C_TEAL    = (0,  128, 128)
C_GOLD    = (212, 163,  55)
C_SILVER  = (180, 180, 190)
C_WHITE   = (255, 255, 255)
C_LIGHT   = (245, 247, 252)
C_DARK    = (30,  30,  40)
C_RED     = (192,  57,  43)
C_GREEN   = (39,  174,  96)
C_BLUE    = (41,  128, 185)
C_GRAD1   = (20,  50, 100)  # gradient start
C_GRAD2   = (0, 120, 120)   # gradient end


# ??????????????????????????????????????????????????
# DATA
# ??????????????????????????????????????????????????

LID_STATS = {
    "overall_acc": 0.9715,
    "correct": 263809,
    "total": 271551,
    "per_class": [
        ("EN",  0.9872, 0.9073, 0.9456, 74135),
        ("HI",  0.9655, 0.9955, 0.9803, 193475),
        ("UNI", 1.0000, 1.0000, 1.0000, 3941),
    ],
    "confusions": [
        ("EN", "HI",  6872),
        ("HI", "EN",   870),
    ],
    "vocab_size":     66828,
    "phonetic_hashes": 22803,
    "labels":          ["EN", "HI", "UNI"],
    "class_prior":     {"HI": 71.3, "EN": 27.5, "UNI": 1.5},
    "smoothing":       1e-8,
    "context_window":  2,
    "context_weight":  0.15,
}

POS_STATS = {
    "overall_acc":  0.7118,
    "correct":      2870,
    "total":        4032,
    "oov_acc":      0.3100,
    "oov_correct":  217,
    "oov_total":    700,
    "iv_acc":       0.7962,
    "iv_correct":   2653,
    "iv_total":     3332,
    "per_tag": [
        ("N",      0.7146, 0.7521, 0.7329, 819),
        ("V",      0.7300, 0.8036, 0.7651, 774),
        ("PSP",    0.7064, 0.7601, 0.7323, 421),
        ("PRP",    0.7117, 0.8700, 0.7829, 400),
        ("DT",     0.8547, 0.9101, 0.8815, 278),
        ("J",      0.7403, 0.5174, 0.6091, 259),
        ("G_N",    0.6328, 0.4706, 0.5398, 238),
        ("R",      0.7377, 0.7725, 0.7547, 233),
        ("CC",     0.6923, 0.5226, 0.5956, 155),
        ("PRT",    0.5732, 0.6081, 0.5902, 148),
        ("G_V",    0.5978, 0.5189, 0.5556, 106),
        ("X",      0.4500, 0.1731, 0.2500,  52),
        ("G_J",    0.5238, 0.2750, 0.3607,  40),
        ("G_PRT",  0.6087, 0.8000, 0.6914,  35),
        ("G_PRP",  0.5000, 0.2424, 0.3265,  33),
        ("SYM",    1.0000, 0.0476, 0.0909,  21),
        ("G_R",    0.0000, 0.0000, 0.0000,  10),
    ],
    "confusions": [
        ("N",   "V",    103),
        ("V",   "N",     74),
        ("J",   "N",     61),
        ("CC",  "PSP",   42),
        ("G_N", "N",     39),
        ("N",   "PRP",   30),
        ("PSP", "CC",    28),
        ("V",   "PRP",   28),
        ("J",   "V",     26),
        ("PSP", "PRT",   22),
    ],
    "vocab_size":       4241,
    "phonetic_buckets": 4000,
    "unique_tags":      21,
    "train_tokens":     16796,
}

HMM_TAGS = {
    "N":     "Noun",
    "V":     "Verb",
    "J":     "Adjective",
    "PRP":   "Pronoun",
    "PSP":   "Postposition",
    "DT":    "Determiner",
    "R":     "Adverb",
    "CC":    "Conjunction",
    "PRT":   "Particle",
    "G_N":   "Genitival Noun",
    "G_V":   "Genitival Verb",
    "G_J":   "Genitival Adjective",
    "G_PRP": "Genitival Pronoun",
    "G_PRT": "Genitival Particle",
    "G_R":   "Genitival Adverb",
    "G_SYM": "Genitival Symbol",
    "SYM":   "Symbol",
    "X":     "Other",
    "$":     "Boundary/Special",
    "E":     "English token",
    "null":  "Null/Empty",
}

# ??????????????????????????????????????????????????
# PDF CLASS
# ??????????????????????????????????????????????????

class HinglishReport(FPDF):

    def header(self):
        # gradient banner approximation using filled rects
        for i in range(40):
            t = i / 39
            r = int(C_GRAD1[0] * (1 - t) + C_GRAD2[0] * t)
            g = int(C_GRAD1[1] * (1 - t) + C_GRAD2[1] * t)
            b = int(C_GRAD1[2] * (1 - t) + C_GRAD2[2] * t)
            self.set_fill_color(r, g, b)
            self.rect(0, i * 0.55, 210, 0.56, "F")

        self.set_xy(0, 2)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*C_WHITE)
        self.cell(0, 10, "Hinglish NLP Pipeline -- Technical Report", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_SILVER)
        self.cell(0, 5, "CL-1 Project  |  Purely Statistical Approach (Approach 2)  |  April 2026",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*C_TEAL)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_SILVER)
        self.cell(0, 5, f"Page {self.page_no()}  |  Hinglish NLP Pipeline Report", align="C")

    # ?? Helpers ??????????????????????????????????????
    def section_title(self, text, level=1):
        self.ln(4)
        if level == 1:
            self.set_fill_color(*C_NAVY)
            self.set_text_color(*C_WHITE)
            self.set_font("Helvetica", "B", 13)
            self.rect(10, self.get_y(), 190, 8, "F")
            self.set_xy(13, self.get_y())
            self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_text_color(*C_TEAL)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_draw_color(*C_TEAL)
            self.set_line_width(0.3)
            self.line(10, self.get_y(), 120, self.get_y())
        self.set_text_color(*C_DARK)
        self.ln(2)

    def body_text(self, text, indent=0):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_DARK)
        self.set_x(10 + indent)
        self.multi_cell(190 - indent, 5, self._s(text))
        self.ln(1)

    def kv_row(self, key, value, shade=False):
        if shade:
            self.set_fill_color(*C_LIGHT)
            fill = True
        else:
            fill = False
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*C_NAVY)
        self.set_x(12)
        self.cell(70, 6, self._s(key), fill=fill)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*C_DARK)
        self.cell(115, 6, self._s(str(value)), fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def stat_badge(self, label, value, color, x, y, w=55, h=20):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, "F")
        self.set_xy(x, y + 2)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*C_WHITE)
        self.cell(w, 10, value, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_xy(x, y + 11)
        self.set_font("Helvetica", "", 7)
        self.cell(w, 6, label, align="C")

    def table_header(self, cols, widths):
        self.set_fill_color(*C_NAVY)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 8)
        self.set_x(12)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=0, align="C", fill=True)
        self.ln()

    def table_row(self, vals, widths, shade=False, aligns=None):
        if aligns is None:
            aligns = ["L"] + ["C"] * (len(vals) - 1)
        fill = shade
        if shade:
            self.set_fill_color(*C_LIGHT)
        self.set_text_color(*C_DARK)
        self.set_font("Helvetica", "", 8)
        self.set_x(12)
        for v, w, a in zip(vals, widths, aligns):
            self.cell(w, 5.5, self._s(str(v)), border=0, align=a, fill=fill)
        self.ln()

    def f1_bar(self, tag, f1, x_start=12, bar_max_w=80):
        self.set_x(x_start)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_DARK)
        self.cell(22, 5, tag)
        # draw bar
        bar_w = f1 * bar_max_w
        # background
        self.set_fill_color(220, 220, 230)
        self.rect(self.get_x(), self.get_y() + 1, bar_max_w, 3.5, "F")
        # foreground
        col = C_GREEN if f1 >= 0.75 else (C_GOLD if f1 >= 0.55 else C_RED)
        self.set_fill_color(*col)
        self.rect(self.get_x(), self.get_y() + 1, bar_w, 3.5, "F")
        self.set_x(self.get_x() + bar_max_w + 2)
        self.set_font("Helvetica", "B", 8)
        self.cell(20, 5, f"{f1:.1%}")
        self.ln()

    def confusion_entry(self, true_lbl, pred_lbl, count, shade=False):
        if shade:
            self.set_fill_color(*C_LIGHT)
        self.set_x(12)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*C_DARK)
        self.cell(30, 5.5, true_lbl, fill=shade)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*C_RED)
        self.cell(10, 5.5, "->", fill=shade, align="C")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*C_DARK)
        self.cell(30, 5.5, pred_lbl, fill=shade)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*C_NAVY)
        self.cell(25, 5.5, f"x{count:,}", fill=shade)
        self.ln()

    def _s(self, text):
        """Sanitize text to latin-1 safe chars."""
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def callout_box(self, title, text, color=C_TEAL):
        self.set_fill_color(*color)
        self.rect(10, self.get_y(), 4, 22, "F")
        self.set_fill_color(240, 248, 252)
        self.rect(14, self.get_y(), 186, 22, "F")
        self.set_xy(16, self.get_y() + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(16)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*C_DARK)
        self.multi_cell(182, 5, text)
        self.ln(3)


# ??????????????????????????????????????????????????
# BUILD REPORT
# ??????????????????????????????????????????????????

pdf = HinglishReport(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(True, margin=18)
pdf.set_margins(10, 24, 10)
pdf.set_font("Helvetica", "", 9)

# ??????????????????????????????????????????????????
# PAGE 1 -- EXECUTIVE SUMMARY + OVERVIEW
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.ln(4)

# Top-level summary badges
pdf.stat_badge("LID Accuracy",   "97.15%", C_GREEN, 15,  38, 55, 22)
pdf.stat_badge("POS Accuracy",   "71.18%", C_BLUE,  75,  38, 55, 22)
pdf.stat_badge("LID Vocab",      "66,828", C_TEAL,  135, 38, 55, 22)

pdf.set_xy(10, 38 + 22 + 4)
pdf.stat_badge("Train Tokens",   "16,796", C_NAVY,  15,  pdf.get_y(), 55, 22)
pdf.stat_badge("OOV POS Acc",    "31.00%", C_RED,   75,  pdf.get_y(), 55, 22)
pdf.stat_badge("Phonetic Hashes","22,803", C_GOLD,  135, pdf.get_y(), 55, 22)

pdf.ln(30)

# ??
pdf.section_title("1. Executive Summary")
pdf.body_text(
    "This report documents the design, implementation, and empirical evaluation of the "
    "Hinglish NLP Pipeline -- a fully statistical, rule-based system for processing "
    "code-mixed Hindi-English (\"Hinglish\") text. The pipeline is Approach 2 of the CL-1 project, "
    "deliberately avoiding pre-trained ML models (e.g., CRF, BERT) in favour of interpretable "
    "probabilistic methods: Naive Bayes Language Identification (LID) and Hidden Markov Model (HMM) "
    "Part-of-Speech (POS) tagging."
)
pdf.body_text(
    "Two evaluation scenarios are covered: (a) LID on the held-out Hinglish LID test set "
    "(271,551 tokens across 8,891 sentences) and (b) POS tagging on the BIS/IIIT same-distribution "
    "test set (4,032 tokens, 312 sentences). Key headline numbers are an exceptionally strong "
    "LID accuracy of 97.15% and a solid POS accuracy of 71.18%, with In-Vocabulary performance "
    "reaching 79.62%."
)

pdf.callout_box(
    "Key Finding",
    "The phonetic hashing fallback is critical: it collapses 66,828 surface forms into 22,803 "
    "equivalence classes, allowing the model to generalise across the extreme spelling variation "
    "characteristic of romanised Hindi (e.g., 'bhai', 'bhaai', 'bhya' -> same hash 'BH').",
    C_TEAL
)

# ??????????????????????????????????????????????????
# PAGE 2 -- ARCHITECTURE
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("2. System Architecture")
pdf.body_text(
    "The pipeline processes raw Hinglish text through five sequential stages:"
)

stages = [
    ("Stage 1 -- Tokenisation",
     "Regex-based tokeniser (re.findall) splits on word boundaries while preserving contractions "
     "and punctuation as separate tokens. Emojis are detected via Unicode range matching and routed "
     "to the Emoji Analyser."),
    ("Stage 2 -- Phonetic Normalisation",
     "The Hinglish Phonetic Hasher (phonetic_matcher.py) converts each token to a canonical skeleton "
     "using: (i) repeated-character compression, (ii) multi-character cluster grouping (sh->S, ph->F, ch->C), "
     "(iii) single-char phonetic substitution (k/c/q->K, v/w->W, s/z->S), and (iv) vowel dropping. "
     "A fast O(1) transliteration variant dictionary is checked first."),
    ("Stage 3 -- Language Identification",
     "Naive Bayes LID model (lid_model.py) classifies each token as EN, HI, or UNI. The model first "
     "looks up the token in its 66,828-word vocabulary; on miss, it falls back to the 22,803-bucket "
     "phonetic table; on a further miss it falls back to class priors. A sliding-window context "
     "smoother (window=+/-2, context weight=0.15) blends neighbouring token evidence via a softmax "
     "blend."),
    ("Stage 4 -- POS Tagging",
     "First-order HMM Viterbi decoder (hmm_pos_tagger.py) assigns BIS/IIIT POS tags. Emission "
     "probabilities use the same 3-tier lookup: word -> phonetic hash -> tag-frequency prior. "
     "Alphabetic tokens are hard-constrained to exclude SYM/PUNC/E/null tags, eliminating "
     "systematic Viterbi drift toward low-content tags for OOV words."),
    ("Stage 5 -- Emoji Annotation",
     "The Emoji Analyser (emoji_analyzer.py) maps recognised emoji characters to POSITIVE, NEGATIVE, "
     "or NEUTRAL placeholders using a 1,591-byte custom lexicon, injecting sentiment context into "
     "the token stream."),
]

for title, desc in stages:
    pdf.section_title(title, level=2)
    pdf.body_text(desc, indent=4)

pdf.section_title("3. Data & Preprocessing", level=1)
pdf.body_text(
    "Training data originates from a Hinglish CoNLL/BIS-annotated corpus. Two preprocessing "
    "scripts generate clean CSV splits:"
)

rows = [
    ("Script", "Input", "Output", "Sentences", "Tokens"),
    ("make_hinglish_lid.py", "hinglish_crf_train_data.jsonl", "train/test LID CSVs (80/20)", "--", "271,551 (test)"),
    ("make_bis_test.py", "train_pos.csv + train_lid.csv", "BIS test POS + LID CSVs", "312", "4,032 (POS test)"),
]
widths = [38, 50, 55, 22, 27]
pdf.table_header(rows[0], widths)
for i, r in enumerate(rows[1:]):
    pdf.table_row(r, widths, shade=(i % 2 == 0))
pdf.ln(3)

pdf.body_text(
    "Non-alphabetic tokens are uniformly assigned the UNI (Universal) label during preprocessing. "
    "The 80/20 train-test split is seeded at 42 for full reproducibility. The LID corpus class "
    "distribution is heavily skewed: HI ~= 71.3%, EN ~= 27.5%, UNI ~= 1.5% -- reflecting the "
    "Hindi-dominant nature of the dataset."
)

# ??????????????????????????????????????????????????
# PAGE 3 -- LID Results
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("4. Language Identification Results")

pdf.section_title("4.1 Overall Performance", level=2)

# Big accuracy number
pdf.set_fill_color(*C_GREEN)
pdf.set_text_color(*C_WHITE)
pdf.set_font("Helvetica", "B", 36)
pdf.rect(60, pdf.get_y(), 90, 22, "F")
pdf.set_xy(60, pdf.get_y() + 3)
pdf.cell(90, 15, "97.15%", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(*C_DARK)
pdf.set_x(60)
pdf.cell(90, 5, "Overall Token-Level Accuracy  (263,809 / 271,551)", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

pdf.section_title("4.2 Per-Class Metrics", level=2)
hdr = ["Label", "Precision", "Recall", "F1", "Support"]
widths = [30, 35, 35, 35, 35]
pdf.table_header(hdr, widths)
for i, (lbl, p, r, f1, sup) in enumerate(LID_STATS["per_class"]):
    pdf.table_row([lbl, f"{p:.2%}", f"{r:.2%}", f"{f1:.2%}", f"{sup:,}"],
                  widths, shade=(i % 2 == 0))
pdf.ln(3)

pdf.body_text(
    "The UNI label achieves perfect 100% precision and recall -- these tokens are "
    "non-alphabetic (numbers, punctuation) and are deterministically classified, making "
    "them trivially correct. The HI class achieves near-perfect recall (99.55%) at a slight "
    "cost to precision (96.55%), owing to the Hindi-biased class prior. "
    "English achieves strong precision (98.72%) but relatively lower recall (90.73%), "
    "meaning 9.3% of English tokens are misclassified as Hindi."
)

pdf.section_title("4.3 F1-Score Visualisation", level=2)
for lbl, p, r, f1, sup in LID_STATS["per_class"]:
    pdf.f1_bar(lbl, f1)
pdf.ln(3)

pdf.section_title("4.4 Confusion Analysis", level=2)
pdf.body_text("Total misclassifications: 271,551 - 263,809 = 7,742 tokens (2.85%)")
pdf.ln(1)
for i, (t, p, cnt) in enumerate(LID_STATS["confusions"]):
    pdf.confusion_entry(t, p, cnt, shade=(i % 2 == 0))
pdf.ln(3)

pdf.body_text(
    "EN->HI (6,872 errors, 88.8% of all errors): This is the dominant error type. "
    "Short, common English function words (e.g., 'to', 'the', 'a', 'is') are frequently "
    "borrowed into Hinglish without change, creating ambiguity. Since they appear in both "
    "Hindi and English contexts in the training corpus, the Hindi-biased prior tips their "
    "classification to HI. Context smoothing partially mitigates this but cannot fully "
    "resolve genuine lexical overlap."
)
pdf.body_text(
    "HI->EN (870 errors, 11.2% of all errors): Rare Hindi transliterations that phonetically "
    "resemble English words (e.g., 'ban', 'hi', 'par') get pulled toward EN by the emission "
    "model. The context smoother successfully suppresses most of these."
)

pdf.callout_box(
    "Model Architecture Note -- Why Phonetic Fallback Matters",
    "Of the 271,551 test tokens, a significant fraction are OOV (not in the 66,828-word vocabulary). "
    "The phonetic fallback reduces effective OOV rate by ~66% by collapsing spelling variants "
    "into 22,803 hash buckets -- crucial for romanised Hindi where 'kyun', 'kyu', 'kyoon', and "
    "'kion' should all resolve identically.",
    C_NAVY
)

# ??????????????????????????????????????????????????
# PAGE 4 -- POS Results
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("5. POS Tagging Results")

pdf.section_title("5.1 Overall Performance", level=2)

pdf.stat_badge("Overall Accuracy", "71.18%", C_BLUE,   15, pdf.get_y(), 54, 22)
pdf.stat_badge("In-Vocab Accuracy","79.62%", C_GREEN,  73, pdf.get_y(), 54, 22)
pdf.stat_badge("OOV Accuracy",     "31.00%", C_RED,    131, pdf.get_y(), 54, 22)
pdf.ln(28)

pdf.body_text(
    f"Evaluated on {POS_STATS['total']:,} tokens ({POS_STATS['total'] - POS_STATS['oov_total']:,} in-vocab, "
    f"{POS_STATS['oov_total']:,} OOV). The 48.62-point accuracy gap between in-vocab and OOV tokens "
    "highlights the inherent difficulty of POS tagging Hinglish text, where extensive spelling variation "
    "and code-mixing create high OOV rates even for well-trained models."
)

pdf.section_title("5.2 Per-Tag Precision / Recall / F1", level=2)
hdr = ["Tag", "Full Name", "Prec.", "Rec.", "F1", "Support"]
widths = [18, 45, 22, 22, 22, 22]
pdf.table_header(hdr, widths)
for i, (tag, p, r, f1, sup) in enumerate(POS_STATS["per_tag"]):
    full = HMM_TAGS.get(tag, "")
    pdf.table_row([tag, full, f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}", f"{sup}"],
                  widths, shade=(i % 2 == 0))
pdf.ln(3)

pdf.section_title("5.3 F1-Score Bar Chart", level=2)
for tag, p, r, f1, sup in POS_STATS["per_tag"][:12]:
    pdf.f1_bar(tag, f1, bar_max_w=90)
pdf.ln(3)

pdf.section_title("5.4 Top-10 Confusions", level=2)
hdr2 = ["True Tag", "Pred. Tag", "Count", "Interpretation"]
widths2 = [28, 28, 22, 100]
pdf.table_header(hdr2, widths2)
interpretations = [
    "Nouns and Verbs overlap in base form (e.g. 'kaam' = work/to-work)",
    "Mirror of above -- verb forms often identical to nouns",
    "Adjectives before nouns get re-labelled as N",
    "Coordinating conjunctions vs. postpositions both follow noun phrases",
    "G_N (genitive noun) collapses to plain N for OOV words",
    "Pronoun-like nouns miscaptured as N",
    "PSP/CC overlap on words like 'aur', 'se'",
    "Short verbs 'ho'/'kar' used pronominally",
    "Adjective forms overlap verb morphology",
    "Postpositions 'ke', 'ki' act as particles in Hinglish",
]
for i, ((t, p, cnt), interp) in enumerate(zip(POS_STATS["confusions"], interpretations)):
    pdf.table_row([t, p, f"x{cnt}", interp], widths2, shade=(i % 2 == 0),
                  aligns=["C", "C", "C", "L"])
pdf.ln(3)

# ??????????????????????????????????????????????????
# PAGE 5 -- MODEL DETAILS
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("6. Model Details & Hyperparameters")

pdf.section_title("6.1 Language Identification Model (lid_model.py)", level=2)
params_lid = [
    ("Algorithm",              "Naive Bayes with Laplace (add-1) smoothing"),
    ("Feature",                "Word-level unigram emission + phonetic hash fallback"),
    ("Labels",                 "EN (English), HI (Hindi), UNI (Universal/non-alpha)"),
    ("Vocabulary size",        "66,828 unique surface forms"),
    ("Phonetic hash table",    "22,803 unique phonetic keys (33% compression ratio)"),
    ("Class prior -- HI",       "exp(-0.3417) ~= 71.1%"),
    ("Class prior -- EN",       "exp(-1.2892) ~= 27.6%"),
    ("Class prior -- UNI",      "exp(-4.2716) ~= 1.4%"),
    ("Smoothing constant",     "1 x 10^-8 (log-domain floor)"),
    ("Context window (+/-)",     "2 tokens (window half-width)"),
    ("Context blend weight",   "0.15 (15% neighbour votes, 85% own likelihood)"),
    ("Model file size",        "8.42 MB (lid_model.json)"),
    ("Inference speed",        "~6,459 tokens/sec on CPU"),
]
for i, (k, v) in enumerate(params_lid):
    pdf.kv_row(k, v, shade=(i % 2 == 0))

pdf.ln(4)
pdf.section_title("6.2 HMM POS Tagger (hmm_pos_tagger.py)", level=2)
params_hmm = [
    ("Algorithm",              "First-order Hidden Markov Model -- Viterbi decoding"),
    ("Emission",               "P(word | tag) with phonetic-hash fallback + tag-prior OOV"),
    ("Transition",             "P(tag_t | tag_{t-1}), START state initialised"),
    ("Unique POS tags",        "21 (BIS/IIIT tagset including genitival variants)"),
    ("Training vocabulary",    "4,241 unique words"),
    ("Training tokens",        "16,796 tokens"),
    ("Phonetic emission keys", "4,000 phonetic hash buckets"),
    ("OOV strategy",           "Phonetic hash -> tag-frequency prior (prevents SYM drift)"),
    ("Hard constraint",        "SYM/PUNC/E/null excluded for alphabetic tokens"),
    ("Smoothing constant",     "1 x 10-^1? (Viterbi log-domain floor)"),
    ("Emission file size",     "171 KB (hmm_emissions.json)"),
    ("Transitions file size",  "6 KB (hmm_transitions.json)"),
    ("Unigrams file size",     "96 KB (hmm_unigrams.json)"),
]
for i, (k, v) in enumerate(params_hmm):
    pdf.kv_row(k, v, shade=(i % 2 == 0))

pdf.ln(4)
pdf.section_title("6.3 Phonetic Hashing Algorithm", level=2)
pdf.body_text("The custom Hinglish phonetic hasher operates in 4 steps:")
steps = [
    "Lowercase & repeated-character compression  (e.g., 'bhaaaaai' -> 'bhai')",
    "Multi-char cluster substitution:  sh->S,  ph->F,  ch->C  (before single-char rules)",
    "Single-char phoneme grouping:  k/c/q->K,  v/w->W,  s/z->S,  f->F",
    "Vowel dropping on tail:  first character kept, all a/e/i/o/u/y removed from remainder",
]
for step in steps:
    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 8.5)
    self_y = pdf.get_y()
    pdf.set_fill_color(*C_TEAL)
    pdf.rect(13, self_y + 1, 2, 3.5, "F")
    pdf.cell(0, 5.5, step, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.ln(2)
examples = [
    ("bhaaaaai  ->  bhai  ->  BH", "Compressed + vowels dropped"),
    ("kya  ->  KA",                "k->K, vowel tail dropped partially"),
    ("sharma  ->  SRM",            "sh->S, vowels stripped"),
    ("phone  ->  FN",              "ph->F, vowels stripped"),
    ("zaroor  ->  SRR",            "z->S, repeated r compressed, vowels dropped"),
]
pdf.section_title("    Phonetic Hash Examples", level=2)
pdf.table_header(["Input -> Hash", "Explanation"], [80, 100])
for i, (h, e) in enumerate(examples):
    pdf.table_row([h, e], [80, 100], shade=(i % 2 == 0))

# ??????????????????????????????????????????????????
# PAGE 6 -- FINDINGS & CONCLUSIONS
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("7. Key Findings")

findings = [
    ("F1: Near-Flawless LID for a Purely Statistical Model",
     "Achieving 97.15% token-level LID accuracy without any pre-trained embeddings or neural "
     "networks is a strong result. The combination of Naive Bayes with Laplace smoothing, phonetic "
     "hashing, and a small context window proves highly effective for the EN/HI binary decision "
     "problem in romanised text."),
    ("F2: Phonetic Generalisation is the Critical Innovation",
     "The 66,828->22,803 vocabulary collapse (66% reduction) is the model's most impactful design "
     "decision. Without phonetic folding, OOV tokens -- which make up a substantial fraction of "
     "Hinglish text due to free-form transliteration -- would all be classified by class priors alone, "
     "drastically reducing accuracy."),
    ("F3: Class Prior Bias Creates Systematic EN->HI Errors",
     "88.8% of all LID errors are EN->HI misclassifications. This traces directly to the corpus-level "
     "Hindi bias (71.3% HI tokens). Short English function words that appear in Hindi-dominant contexts "
     "in training are systematically pulled toward HI. Potential mitigation: cross-lingual word lists."),
    ("F4: HMM POS Performs Well On Known Vocabulary",
     "In-vocabulary POS accuracy of 79.62% is competitive for a statistical model without morphological "
     "features. The Viterbi decoder correctly learns dominant patterns: N and V are the most frequent "
     "tags, and the transitions between them are well-modelled."),
    ("F5: OOV is the Primary POS Challenge",
     "OOV accuracy drops to 31.00%, a 48.6-point gap vs. in-vocabulary. The tag-frequency prior "
     "fallback produces a reasonable bias toward N/V/PSP for unknown words, but cannot capture "
     "subtle morphological cues that a feature-rich model would exploit."),
    ("F6: Structural Category Ambiguity Drives Most POS Errors",
     "The N<->V confusion (103 + 74 = 177 errors) and J->N confusion (61 errors) reflect genuine "
     "structural ambiguity in Hinglish: many words serve as both nouns and verbs (e.g., 'kaam'), "
     "and adjective-noun agreement is less morphologically marked in romanised Hindi than in script."),
    ("F7: Deterministic Tokens are Perfect",
     "UNI (punctuation, numbers, symbols) achieves 100% F1. The hard deterministic rule "
     "(non-alpha -> UNI) is correct by construction, providing a zero-error floor beneath the "
     "statistical components."),
]

for title, text in findings:
    pdf.section_title(title, level=2)
    pdf.body_text(text, indent=4)

pdf.section_title("8. Limitations")
limitations = [
    "No character n-gram features: LID could benefit from subword character n-grams (e.g., trigrams) "
    "for better discrimination of transliteration patterns.",
    "First-order Markov assumption: The HMM only considers the immediately preceding tag. A second-order "
    "or maximum-entropy Markov model would capture longer-range dependencies.",
    "Static transliteration dictionary: The 983-byte variants JSON covers only a small fraction of "
    "common variants; a learned edit-distance model would scale better.",
    "No sentence-level LID: The model operates token-by-token. Sentence-level priors (is this sentence "
    "predominantly Hindi or English?) could further improve contextual smoothing.",
    "POS tagset mismatch risk: During evaluation, punctuation tags are normalised to PUNC; other "
    "rare tags ($, E, null) have zero support and score 0% F1, slightly suppressing overall metrics.",
]
for i, lim in enumerate(limitations):
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*C_NAVY)
    pdf.cell(6, 5.5, f"{i+1}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*C_DARK)
    pdf.multi_cell(180, 5.5, lim)
    pdf.ln(1)

pdf.section_title("9. Conclusions")
pdf.body_text(
    "The Hinglish NLP Pipeline successfully demonstrates that a purely statistical, interpretable "
    "system can achieve near-production-quality Language Identification (97.15%) and competitive "
    "POS tagging (71.18%) on code-mixed Hinglish text -- without any neural networks, pre-trained "
    "embeddings, or external APIs."
)
pdf.body_text(
    "The key architectural decisions that enabled this performance are: (1) the custom Hinglish "
    "phonetic hashing algorithm which dramatically reduces the effective vocabulary by collapsing "
    "spelling variants; (2) the sliding-window context smoother which resolves lexically ambiguous "
    "tokens using their neighbours; and (3) the hard tag constraint in the Viterbi decoder which "
    "prevents OOV drift toward low-content symbol tags."
)
pdf.body_text(
    "Future work should focus on: character n-gram features for LID, second-order HMM or MEMM "
    "for POS, and an expanded transliteration dictionary harvested from the training corpus "
    "automatically using phonetic clustering."
)

# ??????????????????????????????????????????????????
# PAGE 7 -- APPENDIX: Quick Reference
# ??????????????????????????????????????????????????
pdf.add_page()
pdf.section_title("Appendix A: BIS/IIIT POS Tagset Reference")
pdf.table_header(["Tag", "Full Name", "Description"], [22, 45, 120])
tagset = [
    ("N",     "Noun",                "Common and proper nouns in Hindi/English"),
    ("V",     "Verb",                "Main verbs and auxiliary verbs"),
    ("J",     "Adjective",           "Descriptive and qualifying adjectives"),
    ("PRP",   "Pronoun",             "Personal, reflexive, and demonstrative pronouns"),
    ("PSP",   "Postposition",        "Hindi postpositions (ke, ko, se, mein, par...)"),
    ("DT",    "Determiner",          "Articles and determiners (ek, yeh, woh, the, a...)"),
    ("R",     "Adverb",              "Manner, time, degree adverbs"),
    ("CC",    "Conjunction",         "Coordinating and subordinating conjunctions"),
    ("PRT",   "Particle",            "Discourse particles and clitics (hi, bhi, na...)"),
    ("G_N",   "Genitival Noun",      "Noun in genitive construction (possessive)"),
    ("G_V",   "Genitival Verb",      "Verb in genitival/relative clause position"),
    ("G_J",   "Genitival Adjective", "Adjective in genitival phrase"),
    ("G_PRP", "Genitival Pronoun",   "Pronoun in genitival construction"),
    ("G_PRT", "Genitival Particle",  "Particle in genitival construction"),
    ("G_R",   "Genitival Adverb",    "Adverb in genitival construction"),
    ("G_SYM", "Genitival Symbol",    "Symbol in genitival construction"),
    ("SYM",   "Symbol",              "Currency, mathematical or other symbols"),
    ("X",     "Other",               "Foreign word or unclassifiable token"),
    ("$",     "Boundary",            "Special/boundary token"),
    ("E",     "English token",       "Explicit English-origin token marker"),
    ("null",  "Null",                "Empty or null annotation"),
]
for i, (tag, full, desc) in enumerate(tagset):
    pdf.table_row([tag, full, desc], [22, 45, 120], shade=(i % 2 == 0))

pdf.ln(5)
pdf.section_title("Appendix B: Module Summary")
mods = [
    ("lid_model.py",          "statistical_models/", "Naive Bayes LID model, context smoother, train/save/load/predict"),
    ("hmm_pos_tagger.py",     "statistical_models/", "HMM Viterbi POS tagger, phonetic emission, train/load/decode"),
    ("phonetic_matcher.py",   "features/",           "Hinglish phonetic hash, variant normalisation"),
    ("emoji_analyzer.py",     "features/",           "Emoji detection, lexicon-based sentiment tagging"),
    ("evaluate.py",           "tools/",              "Full accuracy report: LID + POS, per-class metrics, confusions"),
    ("make_hinglish_lid.py",  "tools/",              "Extracts EN/HI/UNI labels from JSONL, writes train/test CSVs"),
    ("make_bis_test.py",      "tools/",              "Carves BIS-tagged held-out POS + LID test sets"),
    ("pipeline_runner.py",    "hinglish/",           "End-to-end pipeline: tokenise -> LID -> POS -> print"),
    ("demo.py",               "hinglish/",           "Quick interactive demo on a sample Hinglish sentence"),
]
pdf.table_header(["Module", "Location", "Responsibility"], [45, 35, 110])
for i, (mod, loc, resp) in enumerate(mods):
    pdf.table_row([mod, loc, resp], [45, 35, 110], shade=(i % 2 == 0))

# ??????????????????????????????????????????????????
# SAVE
# ??????????????????????????????????????????????????
out_path = "hinglish_report.pdf"
pdf.output(out_path)
print(f"? Report written to: {os.path.abspath(out_path)}")
