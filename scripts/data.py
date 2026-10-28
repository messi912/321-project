from pathlib import Path

import pandas as pd

from splits import assign_split

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "processed" / "cognate_table.csv"

LANGUAGES = ["hawaiian", "maori", "rarotongan", "samoan", "tahitian", "tongan"]

# Special tokens come first so their IDs never change; <pad> = 0 makes masking easy
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]

# <hawaiian> marks which language a source form is from;
# <to:hawaiian> tells the model which language to produce
SOURCE_TAGS = [f"<{lang}>" for lang in LANGUAGES]
TARGET_TAGS = [f"<to:{lang}>" for lang in LANGUAGES]


def load_examples(table_path=TABLE_PATH):
    """One example per (cognate set, target language), grouped by split."""
    table = pd.read_csv(table_path, index_col="cognateset_id")
    examples = {"train": [], "val": [], "test": []}
    for cognateset_id, row in table.iterrows():
        present = {lang: row[lang] for lang in LANGUAGES if isinstance(row[lang], str)}
        for target in present:
            sources = {lang: form for lang, form in present.items() if lang != target}
            if sources:
                examples[assign_split(cognateset_id)].append({
                    "cognateset_id": cognateset_id,
                    "target_language": target,
                    "target": present[target],
                    "sources": sources,
                })
    return examples


def segment(form):
    """Split a form into the units the model reads. Currently one token per character."""
    return list(form)


class Vocab:
    def __init__(self, characters):
        self.id_to_token = SPECIALS + SOURCE_TAGS + TARGET_TAGS + sorted(characters)
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[PAD]
        self.start_id = self.token_to_id[START]
        self.end_id = self.token_to_id[END]
        self.unk_id = self.token_to_id[UNK]

    @classmethod
    def from_examples(cls, examples):
        # Build from training data only: characters seen only in val/test become <unk>,
        # just as unseen characters would for a real user
        characters = set()
        for example in examples:
            characters.update(segment(example["target"]))
            for form in example["sources"].values():
                characters.update(segment(form))
        return cls(characters)

    def __len__(self):
        return len(self.id_to_token)

    def lookup(self, token):
        return self.token_to_id.get(token, self.unk_id)

    def token(self, token_id):
        return self.id_to_token[token_id]


def encode(example, vocab, source_languages=None):
    """Turn an example into (source IDs, target IDs).

    source: <to:target> <lang> c h a r s <lang> c h a r s ...
    target: <start> c h a r s <end>

    source_languages picks which sources to include and in what order, so training
    can drop or shuffle languages. Defaults to all of them in table order.
    """
    if source_languages is None:
        source_languages = list(example["sources"])

    source = [vocab.lookup(f"<to:{example['target_language']}>")]
    for lang in source_languages:
        source.append(vocab.lookup(f"<{lang}>"))
        source.extend(vocab.lookup(token) for token in segment(example["sources"][lang]))

    target = [vocab.start_id]
    target.extend(vocab.lookup(token) for token in segment(example["target"]))
    target.append(vocab.end_id)
    return source, target


def decode(ids, vocab):
    """Turn target IDs (e.g. model output) back into a form, stopping at <end>."""
    tokens = []
    for token_id in ids:
        if token_id == vocab.end_id:
            break
        token = vocab.token(token_id)
        # Skip <start>, <pad>, and language tags; keep <unk> visible
        if token in SPECIALS and token != UNK or token in SOURCE_TAGS + TARGET_TAGS:
            continue
        tokens.append(token)
    return "".join(tokens)


if __name__ == "__main__":
    examples = load_examples()
    print("Examples per split:", {name: len(split) for name, split in examples.items()})

    vocab = Vocab.from_examples(examples["train"])
    print(f"\nVocabulary size: {len(vocab)}")
    for i, token in enumerate(vocab.id_to_token):
        print(f"  {i:>2}  {token}")

    # Characters in val/test that training never saw will map to <unk>
    for name in ["val", "test"]:
        unseen = set()
        for example in examples[name]:
            for form in [example["target"], *example["sources"].values()]:
                unseen.update(c for c in segment(form) if vocab.lookup(c) == vocab.unk_id)
        print(f"\nUnseen characters in {name}: {sorted(unseen) or 'none'}")

    # Show a few encoded examples
    print("\nEncoded examples:")
    for example in examples["train"][:3]:
        source, target = encode(example, vocab)
        print(f"\n  {example['sources']} -> {example['target_language']}: {example['target']}")
        print("  source tokens:", " ".join(vocab.token(i) for i in source))
        print("  source IDs:   ", source)
        print("  target IDs:   ", target)
        print("  decoded:      ", decode(target, vocab))

    # Round trip: every target must decode back to itself, and every source must
    # contain exactly the original forms' characters after its language tags
    failures = 0
    lengths = {"source": [], "target": []}
    for split in examples.values():
        for example in split:
            source, target = encode(example, vocab)
            lengths["source"].append(len(source))
            lengths["target"].append(len(target))
            source_text = "".join(vocab.token(i) for i in source if vocab.token(i) not in TARGET_TAGS + SOURCE_TAGS)
            if decode(target, vocab) != example["target"] or source_text != "".join(example["sources"].values()):
                failures += 1
    print(f"\nRound-trip failures: {failures} / {sum(len(split) for split in examples.values())}")