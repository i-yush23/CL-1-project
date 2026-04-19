import json

with open('hinglish/data/hmm_unigrams.json', 'r') as f:
    u = json.load(f)
with open('hinglish/data/hmm_transitions.json', 'r') as f:
    t = json.load(f)

words = ['yaar', 'ki', 'hai', 'ek', 'gayi', 'hoon', 'hoga', 'hain', 'jayegi', 'anyway', 'later', 'bye']
print('=== Unigram tag counts per word ===')
for w in words:
    print(f'{w}: {u.get(w, "NOT IN CORPUS")}')

print()
sym_trans = t.get("SYM", {})
print('=== SYM transitions out (top) ===')
print(sorted(sym_trans.items(), key=lambda x: -x[1])[:5])

print()
print('=== Transition TO SYM from each tag (top sources) ===')
sources = {}
for src_tag, targets in t.items():
    if "SYM" in targets:
        sources[src_tag] = targets["SYM"]
print(sorted(sources.items(), key=lambda x: -x[1])[:10])

print()
print('=== All tag counts ===')
emissions = json.load(open('hinglish/data/hmm_emissions.json'))
tag_totals = {tag: len(words_dict) for tag, words_dict in emissions.items()}
print(sorted(tag_totals.items(), key=lambda x: -x[1]))
