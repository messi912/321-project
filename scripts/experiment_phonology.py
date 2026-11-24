import random
from pathlib import Path

import torch
from torch import nn

from data import Vocab, load_examples
from model import Seq2Seq
from metrics import summarize
from train import (
    CONFIG, LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS,
    PATIENCE, SEED, make_batch, evaluate,
    save_checkpoint, load_checkpoint
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "data/models/seq2seq_phonology.pt"

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    examples = load_examples()
    train_examples = list(examples["train"])
    val_examples = examples["val"]

    vocab = Vocab.from_examples(train_examples)

    model = Seq2Seq(len(vocab), vocab.pad_id, **CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # Oversample longer words and phonologically marked targets.
    marked_characters = set("ʔāēīōū")

    def sampling_weight(example):
        target = example["target"]

        length_weight = (
            1 if len(target) <= 4
            else 2 if len(target) <= 7
            else 3
        )

        phonology_weight = (
            1.5 if any(c in marked_characters for c in target)
            else 1.0
        )

        return length_weight * phonology_weight

    weights = [
        sampling_weight(ex)
        for ex in train_examples
    ]

    print("Training examples:", len(train_examples), flush=True)
    print("Validation examples:", len(val_examples), flush=True)
    print("Experimental checkpoint:", EXPERIMENT_PATH, flush=True)

    best_distance = float("inf")
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        sampled = random.choices(
            train_examples,
            weights=weights,
            k=len(train_examples)
        )
        random.shuffle(sampled)

        total_loss = 0.0

        for i in range(0, len(sampled), BATCH_SIZE):
            source, target = make_batch(
                sampled[i:i + BATCH_SIZE],
                vocab,
                augment=True
            )

            logits = model(source, target)
            loss = loss_function(
                logits.reshape(-1, len(vocab)),
                target[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(source)

        val = evaluate(model, vocab, val_examples)
        distance = val["normalized_distance"].mean()
        accuracy = val["exact"].mean()

        improved = distance < best_distance

        if improved:
            best_distance = distance
            best_epoch = epoch
            save_checkpoint(model, vocab, path=EXPERIMENT_PATH)

        if improved or epoch % 10 == 0:
            print(
                f"epoch {epoch:>3} "
                f"loss {total_loss / len(sampled):.3f} "
                f"val acc {accuracy:.3f} "
                f"val NED {distance:.3f}"
                f"{' *' if improved else ''}",
                flush=True
            )

        if epoch - best_epoch >= PATIENCE:
            print("Early stopping.", flush=True)
            break

    best_model, best_vocab = load_checkpoint(EXPERIMENT_PATH)
    results = evaluate(best_model, best_vocab, val_examples)

    print("\n=== LENGTH-BALANCED RESULTS ===")
    print("Best epoch:", best_epoch)
    print("Exact-match accuracy:", round(results["exact"].mean(), 4))
    print("Correct predictions:", int(results["exact"].sum()), "/", len(results))
    print("Normalized edit distance:", round(results["normalized_distance"].mean(), 4))
    print(summarize(results).to_string())

    print("\nOriginal checkpoint: data/models/seq2seq.pt")
    print("Experimental checkpoint:", EXPERIMENT_PATH)

if __name__ == "__main__":
    main()
