import torch
from data import load_examples, encode, decode
from train import load_checkpoint, predict

@torch.no_grad()
def beam_decode(model, vocab, example, width=3, max_len=30):
    model.eval()

    source_ids, _ = encode(example, vocab)
    source = torch.tensor([source_ids], dtype=torch.long)

    outputs, mask, hidden, context = model.encode(source)

    # tokens, log probability, hidden, context, finished
    beams = [([], 0.0, hidden, context, False)]

    forbidden = {
        i for i, token in enumerate(vocab.id_to_token)
        if token.startswith("<") and i != vocab.end_id
    }

    for _ in range(max_len):
        candidates = []

        for tokens, score, h, c, finished in beams:
            if finished:
                candidates.append((tokens, score, h, c, True))
                continue

            previous_id = tokens[-1] if tokens else vocab.start_id
            previous = torch.tensor([previous_id], dtype=torch.long)

            logits, next_h, next_c, _ = model.step(
                previous, h, c, outputs, mask
            )

            log_probs = torch.log_softmax(logits[0], dim=-1)

            for token_id in forbidden:
                log_probs[token_id] = float("-inf")

            values, indices = torch.topk(log_probs, width)

            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append((
                    tokens + [index],
                    score + value,
                    next_h,
                    next_c,
                    index == vocab.end_id
                ))

        candidates.sort(key=lambda b: b[1], reverse=True)
        beams = candidates[:width]

        if all(b[4] for b in beams):
            break

    completed = [b for b in beams if b[4]]
    best = max(completed or beams, key=lambda b: b[1])

    return decode(best[0], vocab)


def main():
    examples = load_examples()["val"]
    model, vocab = load_checkpoint()

    greedy = predict(model, vocab, examples)
    targets = [example["target"] for example in examples]

    baseline_correct = sum(
        p == t for p, t in zip(greedy, targets)
    )

    print(
        f"\nGREEDY: {baseline_correct}/{len(targets)} "
        f"({baseline_correct / len(targets):.2%})",
        flush=True
    )

    for width in [2, 3, 5]:
        print(f"\nTESTING BEAM WIDTH {width}...", flush=True)

        predictions = [
            beam_decode(model, vocab, ex, width)
            for ex in examples
        ]

        correct = sum(
            p == t for p, t in zip(predictions, targets)
        )

        gained = [
            (ex["target_language"], t, g, b)
            for ex, t, g, b in zip(
                examples, targets, greedy, predictions
            )
            if g != t and b == t
        ]

        lost = [
            (ex["target_language"], t, g, b)
            for ex, t, g, b in zip(
                examples, targets, greedy, predictions
            )
            if g == t and b != t
        ]

        print(
            f"ACCURACY: {correct}/{len(targets)} "
            f"({correct / len(targets):.2%})",
            flush=True
        )
        print(
            f"Newly correct: {len(gained)} | "
            f"Regressions: {len(lost)}",
            flush=True
        )

        print("Newly correct examples:")
        for item in gained[:10]:
            print(item)

        print("Regressions:")
        for item in lost[:10]:
            print(item)

    print("\nBEAM SEARCH EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
