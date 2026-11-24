from pathlib import Path
from data import load_examples
from train import load_checkpoint, evaluate

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/models/seq2seq_length.pt"

splits = load_examples()

def signature(example):
    return (
        example["target_language"],
        example["target"],
        tuple(sorted(example["sources"].items()))
    )

train_signatures = {
    signature(ex) for ex in splits["train"]
}

test_examples = splits["test"]

clean_indices = [
    i for i, ex in enumerate(test_examples)
    if signature(ex) not in train_signatures
]

overlap_indices = [
    i for i, ex in enumerate(test_examples)
    if signature(ex) in train_signatures
]

print("=== FINAL TEST EVALUATION ===")
print("Checkpoint:", CHECKPOINT)
print("Total test examples:", len(test_examples))

model, vocab = load_checkpoint(CHECKPOINT)
results = evaluate(model, vocab, test_examples)

clean = results.iloc[clean_indices]
overlap = results.iloc[overlap_indices]

def report(label, frame):
    if len(frame) == 0:
        print(f"{label}: No examples")
        return

    correct = int(frame["exact"].sum())
    accuracy = frame["exact"].mean()
    ned = frame["normalized_distance"].mean()

    print(f"\n{label}")
    print(f"Correct: {correct}/{len(frame)}")
    print(f"Exact-match accuracy: {accuracy:.2%}")
    print(f"Normalized edit distance: {ned:.4f}")

report("FULL TEST SET", results)
report("NON-OVERLAPPING TEST SET", clean)
report("OVERLAPPING TEST EXAMPLES", overlap)

print("\n=== ACCURACY BY LANGUAGE ===")
for language, group in results.groupby("target_language"):
    print(
        f"{language}: "
        f"{int(group['exact'].sum())}/{len(group)} "
        f"({group['exact'].mean():.2%})"
    )

print("\n=== NON-OVERLAPPING ACCURACY BY LANGUAGE ===")
for language, group in clean.groupby("target_language"):
    print(
        f"{language}: "
        f"{int(group['exact'].sum())}/{len(group)} "
        f"({group['exact'].mean():.2%})"
    )

print("\nFINAL TEST EVALUATION COMPLETE")
