import argparse
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
    "dropout": 0.35,
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


def train(seed=SEED, checkpoint_path=CHECKPOINT_PATH, verbose=True):
    """Train one model, keep the epoch with the best val NED, and return its val results."""
    random.seed(seed)
    torch.manual_seed(seed)
    log = print if verbose else (lambda *args, **kwargs: None)

    examples = load_examples()
    vocab = Vocab.from_examples(examples["train"])
    log("Examples per split:", {name: len(split) for name, split in examples.items()})

    model = Seq2Seq(len(vocab), vocab.pad_id, **CONFIG)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
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
            save_checkpoint(model, vocab, checkpoint_path)
        if improved or epoch % 10 == 0:
            log(f"epoch {epoch:>3}  loss {total_loss / len(train_examples):.3f}  "
                f"val acc {val['exact'].mean():.3f}  val NED {distance:.3f}{'  *' if improved else ''}")
        if epoch - best_epoch >= PATIENCE:
            log(f"No improvement for {PATIENCE} epochs, stopping.")
            break

    log(f"\nBest val NED {best_distance:.3f} at epoch {best_epoch}; saved to {checkpoint_path.resolve().relative_to(ROOT)}")
    model, vocab = load_checkpoint(checkpoint_path)
    results = evaluate(model, vocab, examples["val"])
    log("\nVal results for the saved model:")
    log(summarize(results).to_string())
    return results, best_epoch


def train_seeds(n_seeds):
    """Train the same CONFIG with several seeds and report the mean and spread on val.

    One run on 212 val examples can swing a few points by luck alone, so compare
    settings by these averages rather than by single runs.
    """
    print(f"Config: {CONFIG}")
    runs = []
    for seed in range(SEED, SEED + n_seeds):
        path = CHECKPOINT_PATH.with_name(f"seq2seq_seed{seed}.pt")
        results, best_epoch = train(seed, path, verbose=False)
        runs.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "accuracy": results["exact"].mean(),
            "NED": results["normalized_distance"].mean(),
            # Per-language accuracy shows whether one language is driving the difference
            **results.groupby("target_language")["exact"].mean().add_prefix("acc_").to_dict(),
        })
        print(f"seed {seed}  epoch {best_epoch:>3}  val acc {runs[-1]['accuracy']:.3f}  val NED {runs[-1]['NED']:.3f}")

    runs = pd.DataFrame(runs).set_index("seed")
    print("\nPer seed:")
    print(runs.round(3).to_string())
    print("\nMean ± std over seeds:")
    for column in runs.columns:
        print(f"  {column:<16} {runs[column].mean():.3f} ± {runs[column].std():.3f}")
    print(f"\nCheckpoints saved as {CHECKPOINT_PATH.with_name('seq2seq_seed*.pt').resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1,
                        help="train this many seeds and report mean ± std instead of one verbose run")
    args = parser.parse_args()
    if args.seeds > 1:
        train_seeds(args.seeds)
    else:
        train()
