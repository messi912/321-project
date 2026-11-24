from collections import Counter, defaultdict
from data import load_examples

splits = load_examples()
names = ["train", "val", "test"]

print("=== DATASET SIZES ===")
for name in names:
    examples = splits[name]
    print(f"{name}: {len(examples)} examples")

print("\n=== COGNATE SET OVERLAP ===")

sets = {
    name: {
        str(ex["cognateset_id"])
        for ex in splits[name]
        if ex.get("cognateset_id") is not None
    }
    for name in names
}

for i, first in enumerate(names):
    for second in names[i + 1:]:
        overlap = sets[first] & sets[second]
        print(f"{first} vs {second}: {len(overlap)} shared cognate sets")
        if overlap:
            print("Examples:", sorted(overlap)[:10])

print("\n=== REPEATED TARGET FORMS ===")

for name in names:
    pairs = Counter(
        (ex["target_language"], ex["target"])
        for ex in splits[name]
    )

    duplicates = {
        pair: count
        for pair, count in pairs.items()
        if count > 1
    }

    print(f"\n{name}: {len(duplicates)} repeated language/form pairs")

    for pair, count in sorted(
        duplicates.items(),
        key=lambda item: -item[1]
    )[:10]:
        print(f"  {pair}: {count} occurrences")

print("\n=== LANGUAGE DISTRIBUTION ===")

for name in names:
    counts = Counter(
        ex["target_language"]
        for ex in splits[name]
    )

    print(f"\n{name}:")
    for language, count in sorted(counts.items()):
        print(f"  {language}: {count}")

print("\n=== CROSS-SPLIT EXACT EXAMPLE OVERLAP ===")

def signature(example):
    return (
        example["target_language"],
        example["target"],
        tuple(sorted(example["sources"].items()))
    )

signatures = {
    name: {signature(ex) for ex in splits[name]}
    for name in names
}

for i, first in enumerate(names):
    for second in names[i + 1:]:
        overlap = signatures[first] & signatures[second]
        print(f"{first} vs {second}: {len(overlap)} identical examples")

print("\nDATA AUDIT COMPLETE")
