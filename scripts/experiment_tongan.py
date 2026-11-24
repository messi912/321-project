from pathlib import Path
from collections import Counter

from data import load_examples
from train import load_checkpoint, evaluate

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/models/seq2seq_length.pt"

model, vocab = load_checkpoint(CHECKPOINT)
examples = load_examples()["val"]
results = evaluate(model, vocab, examples)

tongan = results[results["target_language"] == "tongan"].copy()
errors = tongan[~tongan["exact"].astype(bool)].copy()

print("\n=== TONGAN PERFORMANCE ===")
print("Examples:", len(tongan))
print("Correct:", int(tongan["exact"].sum()))
print("Errors:", len(errors))
print("Accuracy:", f"{tongan['exact'].mean():.2%}")

print("\n=== ERROR TYPES ===")

categories = Counter()

for _, row in errors.iterrows():
    target = row["target"]
    prediction = row["prediction"]

    if len(prediction) < len(target):
        categories["Prediction too short"] += 1
    elif len(prediction) > len(target):
        categories["Prediction too long"] += 1
    else:
        categories["Same length, wrong characters"] += 1

    if "ʔ" in target or "ʔ" in prediction:
        categories["Glottal stop involved"] += 1

    if any(c in target + prediction for c in "āēīōū"):
        categories["Long vowel/macron involved"] += 1

for category, count in categories.items():
    print(f"{category}: {count}")

print("\n=== ALL TONGAN ERRORS ===")

for _, row in errors.iterrows():
    print(
        f"Actual: {row['target']:<15} "
        f"Predicted: {row['prediction']:<15} "
        f"Normalized distance: {row['normalized_distance']:.3f}"
    )

print("\n=== ERRORS BY NUMBER OF SOURCE LANGUAGES ===")

for count in sorted({
    len(sources) for sources in tongan["sources"]
}):
    subset = tongan[
        tongan["sources"].apply(len) == count
    ]
    print(
        f"{count} source languages: "
        f"{int(subset['exact'].sum())}/{len(subset)} correct "
        f"({subset['exact'].mean():.2%})"
    )

print("\nTONGAN DIAGNOSTIC COMPLETE")
