import random
from pathlib import Path

import torch
from torch import nn

from data import Vocab, encode, load_examples
from train import (
    CONFIG, LEARNING_RATE, BATCH_SIZE,
    MAX_EPOCHS, PATIENCE, SEED,
    evaluate, pad, save_checkpoint, load_checkpoint
)
from model import Seq2Seq

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/models/seq2seq_fullsources.pt"


def make_mixed_batch(examples, vocab):
    sources, targets = [], []

    for example in examples:
        languages = list(example["sources"])

        if random.random() >= 0.70:
            languages = random.sample(
                languages, random.randint(1, len(languages))
            )

        source, target = encode(example, vocab, languages)
        sources.append(source)
        targets.append(target)

    return pad(sources, vocab.pad_id), pad(targets, vocab.pad_id)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    splits = load_examples()
    train_examples = list(splits["train"])
    val_examples = splits["val"]

    vocab = Vocab.from_examples(train_examples)
    model = Seq2Seq(len(vocab), vocab.pad_id, **CONFIG)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    weights = [
        1 if len(ex["target"]) <= 4
        else 2 if len(ex["target"]) <= 7
        else 3
        for ex in train_examples
    ]

    best_distance = float("inf")
    best_epoch = 0

    print("Training examples:", len(train_examples), flush=True)
    print("Validation examples:", len(val_examples), flush=True)
    print("Experimental checkpoint:", CHECKPOINT, flush=True)

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
            batch = sampled[i:i + BATCH_SIZE]
            source, target = make_mixed_batch(batch, vocab)

            logits = model(source, target)
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                target[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch)

        val = evaluate(model, vocab, val_examples)
        distance = val["normalized_distance"].mean()
        accuracy = val["exact"].mean()

        if distance < best_distance:
            best_distance = distance
            best_epoch = epoch
            save_checkpoint(model, vocab, CHECKPOINT)

        if epoch % 10 == 0 or epoch == best_epoch:
            print(
                f"epoch {epoch:>3} "
                f"loss {total_loss / len(sampled):.3f} "
                f"val acc {accuracy:.3f} "
                f"val NED {distance:.3f}",
                flush=True
            )

        if epoch - best_epoch >= PATIENCE:
            print("Early stopping.", flush=True)
            break

    best_model, best_vocab = load_checkpoint(CHECKPOINT)
    final = evaluate(best_model, best_vocab, val_examples)

    print("\n=== FULL-SOURCE TRAINING RESULTS ===")
    print("Best epoch:", best_epoch)
    print("Correct:", int(final["exact"].sum()), "/", len(final))
    print("Accuracy:", f"{final['exact'].mean():.2%}")
    print("Normalized edit distance:",
          round(final["normalized_distance"].mean(), 4))
    print(final.groupby("target_language")["exact"].agg(
        ["count", "mean"]
    ).to_string())

    print("\nOriginal checkpoint remains unchanged.")
    print("FULL-SOURCE EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
