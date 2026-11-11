import sys

from data import load_examples
from metrics import summarize
from train import evaluate, load_checkpoint

# Score the saved model the same way baseline.py scores the baselines.
# Defaults to val; pass "test" only for final numbers so test doesn't steer tuning:
#   python -X utf8 scripts/evaluate.py test

split = sys.argv[1] if len(sys.argv) > 1 else "val"

model, vocab = load_checkpoint()
results = evaluate(model, vocab, load_examples()[split])

print(f"===== {split.upper()} =====")
print(summarize(results).to_string())

print(f"\nSample errors ({split}):")
errors = results[~results["exact"]]
print(errors[["target_language", "target", "prediction", "sources"]].head(20).to_string(index=False))
