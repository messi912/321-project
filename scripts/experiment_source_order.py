import random
from pathlib import Path

import torch

from data import load_examples, encode, decode
from train import load_checkpoint, pad

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/models/seq2seq_length.pt"

model, vocab = load_checkpoint(CHECKPOINT)
model.eval()

examples = load_examples()["val"]


def predict_with_order(example, languages):
    source_ids, _ = encode(example, vocab, languages)
    source = pad([source_ids], vocab.pad_id)

    with torch.no_grad():
        output = model.greedy_decode(
            source,
            vocab.start_id,
            vocab.end_id
        )

    return decode(output[0].tolist(), vocab)


def get_orders(example):
    languages = list(example["sources"])

    orders = [
        languages,
        list(reversed(languages)),
        sorted(languages),
        sorted(languages, reverse=True)
    ]

    rng = random.Random(321)
    for _ in range(6):
        shuffled = languages.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)

    # Remove duplicate permutations.
    unique = []
    seen = set()

    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique.append(order)

    return unique


total = len(examples)
baseline_correct = 0
stable = 0
unstable = 0
any_order_correct = 0
majority_correct = 0

by_language = {}

print("Evaluating source-order sensitivity...", flush=True)

for example in examples:
    target = example["target"]
    language = example["target_language"]

    orders = get_orders(example)
    predictions = [
        predict_with_order(example, order)
        for order in orders
    ]

    baseline = predictions[0]

    # Deterministic majority vote, breaking ties
    # in favor of the earliest encountered prediction.
    majority = max(
        dict.fromkeys(predictions),
        key=predictions.count
    )

    baseline_correct += baseline == target
    majority_correct += majority == target
    any_order_correct += target in predictions

    changed = len(set(predictions)) > 1

    stable += not changed
    unstable += changed

    if language not in by_language:
        by_language[language] = {
            "total": 0,
            "unstable": 0,
            "baseline_correct": 0,
            "majority_correct": 0
        }

    stats = by_language[language]
    stats["total"] += 1
    stats["unstable"] += changed
    stats["baseline_correct"] += baseline == target
    stats["majority_correct"] += majority == target


print("\n=== SOURCE-ORDER RESULTS ===")
print(f"Validation examples: {total}")
print(f"Stable predictions: {stable}/{total} ({stable/total:.2%})")
print(f"Changed predictions: {unstable}/{total} ({unstable/total:.2%})")

print("\n=== ACCURACY ===")
print(f"Original order: {baseline_correct}/{total} ({baseline_correct/total:.2%})")
print(f"Majority vote: {majority_correct}/{total} ({majority_correct/total:.2%})")
print(f"Any order correct (oracle only): {any_order_correct}/{total} ({any_order_correct/total:.2%})")

print("\n=== BY TARGET LANGUAGE ===")
for language, stats in sorted(by_language.items()):
    n = stats["total"]
    print(
        f"{language}: "
        f"changed={stats['unstable']}/{n}, "
        f"original={stats['baseline_correct']}/{n}, "
        f"majority={stats['majority_correct']}/{n}"
    )

print("\nSOURCE-ORDER EXPERIMENT COMPLETE")
