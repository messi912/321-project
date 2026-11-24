from pathlib import Path

import torch

from data import load_examples, encode, decode
from train import load_checkpoint, predict

ROOT = Path(__file__).resolve().parents[1]

@torch.no_grad()
def ensemble_predict(models, vocab, example, weights):
    source_ids, _ = encode(example, vocab)
    source = torch.tensor([source_ids], dtype=torch.long)

    states = []
    for model in models:
        model.eval()
        outputs, mask, hidden, context = model.encode(source)
        states.append((outputs, mask, hidden, context))

    previous = torch.tensor([vocab.start_id])
    predictions = []

    for _ in range(30):
        probabilities = []
        next_states = []

        for model, state in zip(models, states):
            outputs, mask, hidden, context = state
            logits, hidden, context, _ = model.step(
                previous, hidden, context, outputs, mask
            )
            probabilities.append(torch.softmax(logits, dim=-1))
            next_states.append((outputs, mask, hidden, context))

        combined = sum(
            weight * probability
            for weight, probability in zip(weights, probabilities)
        )

        next_id = int(combined.argmax(dim=-1).item())
        predictions.append(next_id)

        if next_id == vocab.end_id:
            break

        previous = torch.tensor([next_id])
        states = next_states

    return decode(predictions, vocab)


def main():
    examples = load_examples()["val"]

    original, vocab_original = load_checkpoint(
        ROOT / "data/models/seq2seq.pt"
    )
    balanced, vocab_balanced = load_checkpoint(
        ROOT / "data/models/seq2seq_length.pt"
    )

    assert vocab_original.id_to_token == vocab_balanced.id_to_token, (
        "Model vocabularies differ; cannot ensemble safely."
    )

    vocab = vocab_original
    targets = [ex["target"] for ex in examples]

    print("Validation examples:", len(examples), flush=True)

    for name, model in [
        ("Original", original),
        ("Length-balanced", balanced),
    ]:
        predictions = predict(model, vocab, examples)
        correct = sum(p == t for p, t in zip(predictions, targets))
        print(
            f"{name}: {correct}/{len(examples)} "
            f"({correct / len(examples):.2%})",
            flush=True
        )

    for original_weight in [0.5, 0.25, 0.1]:
        weights = [original_weight, 1 - original_weight]

        predictions = [
            ensemble_predict(
                [original, balanced], vocab, example, weights
            )
            for example in examples
        ]

        correct = sum(p == t for p, t in zip(predictions, targets))

        print(
            f"Ensemble {weights}: "
            f"{correct}/{len(examples)} "
            f"({correct / len(examples):.2%})",
            flush=True
        )

    print("\nENSEMBLE EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
