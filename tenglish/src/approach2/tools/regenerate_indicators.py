"""
regenerate_indicators.py
Generates te_indicators.json and en_indicators.json from scratch.

te_indicators: Romanized Telugu words that are strong Telugu-language signals.
en_indicators: English words (common vocab) that are strong English-language signals.

Run this once to produce the JSON files used by CodeswitchSegmenter.
Usage:
    python regenerate_indicators.py
"""

import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Telugu indicators — Romanized Telugu words highly unlikely to be English
# ---------------------------------------------------------------------------
TE_INDICATORS = [
    # Pronouns
    "nuvvu", "naaku", "meeru", "vaadu", "aame", "memu", "manamu", "vaallu", "neeku",
    # Postpositions / case markers
    "tho", "lo", "ki", "ku", "nunchi", "varaku", "kosam", "gurinchi", "meeda",
    "dwara", "valla", "veru", "kinda", "paina",
    # Common verbs (infinitive / conjugated)
    "cheyyi", "chestaa", "chestunnaa", "chesaanu", "chesaadu", "chesindi",
    "undi", "undhi", "ledu", "untundi", "untaanu", "untaadu",
    "vastaanu", "vastaadu", "vastundi", "vellanu", "velladu", "vellindi",
    "cheppanu", "cheppadu", "cheppindi", "cheppu", "cheppandi",
    "tinnanu", "tinnadu", "tinnindi", "tinu", "tinandi",
    "padukunnanu", "padukunnadu", "padukundu",
    "aadutunnanu", "aadutunnadu", "aadutunnindi",
    "chudanu", "chudadu", "chudandi", "chuu",
    "vinaanu", "vinaadu", "vinaandi", "vinu",
    "raasetaanu", "raasetaadu", "raasindi",
    "nerchukovaali", "nerchukunnaanu",
    "theliyadu", "theliyaledu", "thelisindi",
    "kopanga", "naccindi", "naccaledu",
    # Auxiliaries / particles
    "kaadu", "kadu", "ani", "ayithe", "aythe", "aithe", "ante", "anta",
    "okka", "oka", "anni", "kooda", "kuda", "malli", "ippudu", "eppudu",
    "ekkada", "ela", "emi", "enni", "enduku", "evadu", "evari", "edhi",
    "konchem", "chaala", "baaga", "miku", "mee",
    # Interjections / discourse markers
    "ayyo", "arre", "abba", "ra", "raa", "da", "daa", "ba", "baa",
    "anna", "akka", "annaa", "ayyaa", "ammaa", "naanna",
    "sare", "sarrey", "okay", "opika", "chalu", "chaaluuu",
    "super", "baagundi", "manchidi",
    # Question words
    "em", "emi", "ela", "eppudu", "ekkada", "evadu", "evari", "edhi", "enduku", "enni",
    # Negative
    "ledu", "kaadu", "venukki", "vaddu", "vaddhu",
    # Adverbs / adjectives
    "taruvata", "mundu", "ippudu", "inkaa", "maatram", "kooda",
    "chinnaga", "peddhaga", "manchiga", "cheduga",
    # Common nouns (highly TE-specific)
    "illu", "ooru", "pani", "manishi", "pilladu", "pillaa", "ammayi", "abbaayi",
    "naalugu", "aidu", "aaru", "edu", "enimidi", "tommidi", "padhi",
    "povu", "raavu", "tinadam", "cheyyadam", "maatlaadadam",
]

# ---------------------------------------------------------------------------
# English indicators — common English words that are strong EN signals
# ---------------------------------------------------------------------------
EN_INDICATORS = [
    # Function words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "to", "of", "in", "on", "at",
    "by", "for", "with", "about", "from", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "because", "as", "until",
    "while", "although", "if", "or", "but", "and", "not", "no", "nor", "so",
    "yet", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom",
    # Common content words
    "time", "year", "people", "way", "day", "man", "woman", "child",
    "work", "life", "hand", "part", "place", "case", "week", "company",
    "system", "program", "question", "government", "number", "night",
    "point", "home", "water", "room", "mother", "area", "money", "story",
    "fact", "month", "lot", "right", "study", "book", "eye", "job",
    "word", "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "among",
    "ever", "stand", "own", "move", "live", "happen", "carry", "talk",
    "run", "bring", "seem", "leave", "want", "look", "love", "become",
    "lead", "keep", "feel", "begin", "show", "hear", "play", "start",
    "never", "last", "always", "something", "nothing", "everything",
    # Social-media / informal English
    "lol", "omg", "tbh", "ngl", "idk", "imo", "btw", "wbu", "hbu",
    "brb", "afk", "irl", "fyi", "smh", "ikr", "rn", "fr", "nvm",
    "same", "mood", "vibe", "literally", "basically", "honestly",
    "actually", "seriously", "totally", "absolutely", "definitely",
    "okay", "ok", "yeah", "yep", "nope", "sorry", "please", "thanks",
    "thank", "hi", "hello", "hey", "bye", "yes", "no",
]

# ---------------------------------------------------------------------------
# Write out
# ---------------------------------------------------------------------------
te_path = os.path.join(OUTPUT_DIR, "te_indicators.json")
en_path = os.path.join(OUTPUT_DIR, "en_indicators.json")

with open(te_path, "w", encoding="utf-8") as f:
    json.dump(list(set(TE_INDICATORS)), f, ensure_ascii=False, indent=2)
print(f"Saved {len(set(TE_INDICATORS))} Telugu indicators → {te_path}")

with open(en_path, "w", encoding="utf-8") as f:
    json.dump(list(set(EN_INDICATORS)), f, ensure_ascii=False, indent=2)
print(f"Saved {len(set(EN_INDICATORS))} English indicators → {en_path}")
