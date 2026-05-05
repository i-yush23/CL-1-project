from collections import Counter

# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(file_path):
    sentences = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        sent = []
        tags = []

        for line in f:
            line = line.strip()

            if line == "":
                if sent:
                    sentences.append(sent)
                    labels.append(tags)
                    sent, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                sent.append(parts[0])
                tags.append(parts[1])

        if sent:
            sentences.append(sent)
            labels.append(tags)

    return sentences, labels


def flatten(x):
    return [i for sub in x for i in sub]


# ----------------------------
# SIMPLE METRICS (NO SKLEARN)
# ----------------------------
def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def tag_wise_accuracy(y_true, y_pred):
    correct = Counter()
    total = Counter()

    for g, p in zip(y_true, y_pred):
        total[g] += 1
        if g == p:
            correct[g] += 1

    return {tag: correct[tag] / total[tag] for tag in total}


# ----------------------------
# ANALYSER
# ----------------------------
def evaluate(model, test_file):
    sentences, gold = load_data(test_file)

    preds = []
    errors = []

    for sent, g in zip(sentences, gold):
        p = model.predict(sent)
        preds.append(p)

        for w, gt, pr in zip(sent, g, p):
            if gt != pr:
                errors.append((w, gt, pr))

    y_true = flatten(gold)
    y_pred = flatten(preds)

    print("\n===== RESULTS =====\n")

    print("Token Accuracy:", accuracy(y_true, y_pred))

    # sentence accuracy
    correct_sent = sum(1 for g, p in zip(gold, preds) if g == p)
    print("Sentence Accuracy:", correct_sent / len(gold))

    print("\n===== TAG-WISE ACCURACY =====\n")
    tag_acc = tag_wise_accuracy(y_true, y_pred)
    for tag, acc in tag_acc.items():
        print(f"{tag}: {acc:.4f}")

    print("\n===== SAMPLE ERRORS =====\n")
    for e in errors[:20]:
        print(f"{e[0]} | GOLD: {e[1]} | PRED: {e[2]}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    from models.hmm_pos_tagger import HMMPOSTagger

    model = HMMPOSTagger()
    model.train("data/train.txt")   # adjust path

    evaluate(model, "data/telugu-test.txt")