"""Combine several trained models into one predictor (an ensemble).

Each model was trained with a different seed, so each makes slightly different
mistakes. At every letter, the ensemble averages the models' probabilities and
keeps the best few partial words (beam search). This gives a more reliable top
answer plus ranked alternatives with a confidence score.

Train the models first (one per seed), for example:
    SEQ2SEQ_SEED=1 SEQ2SEQ_CHECKPOINT=../data/models/seq2seq_1.pt python3 train.py
Then score them together:
    python3 ensemble.py test
"""
import math
import sys
from pathlib import Path

import pandas as pd
import torch

from data import decode, load_examples
from metrics import score, summarize
from train import load_checkpoint, make_batch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
BEAM_SIZE = 5


def load_models(paths=None):
    """Load every seq2seq_<seed>.pt checkpoint. All share one vocabulary."""
    paths = paths or sorted(MODEL_DIR.glob("seq2seq_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No seq2seq_*.pt checkpoints in {MODEL_DIR}")
    loaded = [load_checkpoint(path) for path in paths]
    models = [model for model, _ in loaded]
    vocab = loaded[0][1]
    return models, vocab


@torch.no_grad()
def predict_top(models, vocab, example, beam_size=BEAM_SIZE, max_len=30):
    """Return up to beam_size (word, confidence) pairs for one example, best first."""
    source, _ = make_batch([example], vocab)
    encoded = [model.encode(source) for model in models]

    # Each beam: (total log probability, letter IDs so far, each model's state, finished?)
    start_states = [(hidden, context) for _, _, hidden, context in encoded]
    beams = [(0.0, [], start_states, False)]

    for _ in range(max_len):
        if all(finished for *_, finished in beams):
            break
        candidates = []
        for log_prob, ids, states, finished in beams:
            if finished:
                candidates.append((log_prob, ids, states, True))
                continue

            previous = torch.tensor([ids[-1] if ids else vocab.start_id])
            step_log_probs, new_states = [], []
            for model, (outputs, mask, _, _), (hidden, context) in zip(models, encoded, states):
                logits, hidden, context, _ = model.step(previous, hidden, context, outputs, mask)
                step_log_probs.append(torch.log_softmax(logits, dim=-1))
                new_states.append((hidden, context))

            # The ensemble: average the models' next-letter probabilities
            average = torch.stack(step_log_probs).mean(dim=0)[0]
            top = torch.topk(average, beam_size)
            for value, token in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((log_prob + value, ids + [token], new_states, token == vocab.end_id))

        # Rank by average log probability per letter so short words aren't favoured
        candidates.sort(key=lambda candidate: candidate[0] / len(candidate[1]), reverse=True)
        beams = candidates[:beam_size]

    results = []
    for log_prob, ids, _, _ in beams:
        word = decode(ids, vocab)
        confidence = math.exp(log_prob / len(ids))
        if word not in [w for w, _ in results]:
            results.append((word, round(confidence, 3)))
    return results


def evaluate_ensemble(models, vocab, examples):
    rows = []
    for example in examples:
        top = predict_top(models, vocab, example)
        prediction = top[0][0]
        rows.append({
            "target_language": example["target_language"],
            "sources": example["sources"],
            "target": example["target"],
            "prediction": prediction,
            "in_top_3": example["target"] in [word for word, _ in top[:3]],
            **score(example["target"], prediction),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    models, vocab = load_models()
    results = evaluate_ensemble(models, vocab, load_examples()[split])

    print(f"===== {split.upper()} ({len(models)}-model ensemble) =====")
    print(summarize(results).to_string())
    print(f"\nCorrect word in top 3: {results['in_top_3'].mean():.3f}")