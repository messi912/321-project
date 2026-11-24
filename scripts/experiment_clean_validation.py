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

validation = splits["val"]

clean_indices = [
    i for i, ex in enumerate(validation)
    if signature(ex) not in train_signatures
]

overlap_indices = [
    i for i, ex in enumerate(validation)
    if signature(ex) in train_signatures
]

model, vocab = load_checkpoint(CHECKPOINT)
results = evaluate(model, vocab, validation)

clean = results.iloc[clean_indices]
overlap = results.iloc[overlap_indices]

print("=== VALIDATION CONTAMINATION CHECK ===")
print("Total examples:", len(validation))
print("Overlapping rows:", len(overlap))
print("Non-overlapping rows:", len(clean))

print("\n=== ACCURACY ===")
print(
    "Full validation:",
    f"{int(results['exact'].sum())}/{len(results)}",
    f"({results['exact'].mean():.2%})"
)

print(
    "Non-overlapping validation:",
    f"{int(clean['exact'].sum())}/{len(clean)}",
    f"({clean['exact'].mean():.2%})"
)

if len(overlap):
    print(
        "Overlapping examples:",
        f"{int(overlap['exact'].sum())}/{len(overlap)}",
        f"({overlap['exact'].mean():.2%})"
    )

print("\nCLEAN VALIDATION CHECK COMPLETE")
