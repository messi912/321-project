import random
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from data import Vocab, decode, encode, load_examples
from metrics import score, summarize
from model import Seq2Seq

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = Path(os.environ.get(
    "SEQ2SEQ_CHECKPOINT",
    str(ROOT / "data" / "models" / "seq2seq.pt")
))

CONFIG = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "dropout": 0.3,
}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 300
# Stop once val normalized edit distance hasn't improved for this many epochs
PATIENCE = 30
SEED = 321


def pad(sequences, pad_id):
    longest = max(len(sequence) for sequence in sequences)
    return torch.tensor([sequence + [pad_id] * (longest - len(sequence)) for sequence in sequences])


def make_batch(examples, vocab, augment=False):
    """Encode and pad a list of examples into (source, target) tensors.

    With augment=True each example keeps a random non-empty subset of its source
    languages in a random order. Real cognate sets are missing languages all the time,
    so this teaches the model to work from whatever sisters it is given, and it turns
    one example into many different inputs.
    """
    sources, targets = [], []
    for example in examples:
        languages = list(example["sources"])
        if augment:
            languages = random.sample(languages, random.randint(1, len(languages)))
        source, target = encode(example, vocab, languages)
        sources.append(source)
        targets.append(target)
    return pad(sources, vocab.pad_id), pad(targets, vocab.pad_id)


def predict(model, vocab, examples):
    """Greedy predictions for a list of examples, using all their source languages."""
    model.eval()
    predictions = []
    for i in range(0, len(examples), BATCH_SIZE):
        source, _ = make_batch(examples[i:i + BATCH_SIZE], vocab)
        output = model.greedy_decode(source, vocab.start_id, vocab.end_id)
        predictions.extend(decode(ids, vocab) for ids in output.tolist())
    return predictions


def evaluate(model, vocab, examples):
    rows = []
    for example, prediction in zip(examples, predict(model, vocab, examples)):
        rows.append({
            "target_language": example["target_language"],
            "sources": example["sources"],
            "target": example["target"],
            "prediction": prediction,
            **score(example["target"], prediction),
        })
    return pd.DataFrame(rows)


def save_checkpoint(model, vocab, path=CHECKPOINT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": CONFIG,
        "tokens": vocab.id_to_token,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path=CHECKPOINT_PATH):
    checkpoint = torch.load(path)
    # Vocab sorts its characters, so rebuilding from the saved characters gives the same IDs
    characters = [token for token in checkpoint["tokens"] if not token.startswith("<")]
    vocab = Vocab(characters)
    assert vocab.id_to_token == checkpoint["tokens"]
    model = Seq2Seq(len(vocab), vocab.pad_id, **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    examples = load_examples()
    vocab = Vocab.from_examples(examples["train"])
    print("Examples per split:", {name: len(split) for name, split in examples.items()})

    model = Seq2Seq(len(vocab), vocab.pad_id, **CONFIG)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    best_distance, best_epoch = float("inf"), 0
    train_examples = list(examples["train"])
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        random.shuffle(train_examples)
        total_loss = 0.0
        for i in range(0, len(train_examples), BATCH_SIZE):
            source, target = make_batch(train_examples[i:i + BATCH_SIZE], vocab, augment=True)
            logits = model(source, target)
            loss = loss_function(logits.reshape(-1, len(vocab)), target[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(source)

        val = evaluate(model, vocab, examples["val"])
        distance = val["normalized_distance"].mean()
        improved = distance < best_distance
        if improved:
            best_distance, best_epoch = distance, epoch
            save_checkpoint(model, vocab)
        if improved or epoch % 10 == 0:
            print(f"epoch {epoch:>3}  loss {total_loss / len(train_examples):.3f}  "
                  f"val acc {val['exact'].mean():.3f}  val NED {distance:.3f}{'  *' if improved else ''}")
        if epoch - best_epoch >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping.")
            break

    print(f"\nBest val NED {best_distance:.3f} at epoch {best_epoch}; saved to {CHECKPOINT_PATH.resolve().relative_to(ROOT)}")
    model, vocab = load_checkpoint()
    print("\nVal results for the saved model:")
    print(summarize(evaluate(model, vocab, examples["val"])).to_string())


if __name__ == "__main__":
    train()
