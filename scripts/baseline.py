from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from splits import assign_split

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "processed" / "cognate_table.csv"

LANGUAGES = ["hawaiian", "maori", "rarotongan", "samoan", "tahitian", "tongan"]

# Task: for each cognate set, hide one language's form and predict it from the others.
# Two baselines:
#   copy   - copy the form from the most similar available language
#   mapped - same, but first apply character correspondences learned from training data
#            (e.g. Hawaiian k -> Maori t)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def align(source, target):
    """Levenshtein alignment as (source_char, target_char) pairs; "" marks a gap."""
    rows, cols = len(source) + 1, len(target) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
            )

    pairs = []
    i, j = len(source), len(target)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (source[i - 1] != target[j - 1]):
            pairs.append((source[i - 1], target[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((source[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", target[j - 1]))
            j -= 1
    return pairs[::-1]


def make_examples(table):
    """One example per (cognate set, target language) with at least one other language present."""
    examples = []
    for cognateset_id, row in table.iterrows():
        present = {lang: row[lang] for lang in LANGUAGES if isinstance(row[lang], str)}
        for target in present:
            sources = {lang: form for lang, form in present.items() if lang != target}
            if sources:
                examples.append({
                    "cognateset_id": cognateset_id,
                    "target_language": target,
                    "target": present[target],
                    "sources": sources,
                })
    return examples


def learn_mappings(train):
    """Count source -> target character correspondences for every language pair."""
    counts = defaultdict(lambda: defaultdict(Counter))
    for example in train:
        for source_lang, form in example["sources"].items():
            pair = (source_lang, example["target_language"])
            for source_char, target_char in align(form, example["target"]):
                # Insertions have no source character to attach to, so skip them
                if source_char:
                    counts[pair][source_char][target_char] += 1
    return {
        pair: {char: options.most_common(1)[0][0] for char, options in chars.items()}
        for pair, chars in counts.items()
    }


def apply_mapping(form, mapping):
    # Characters never seen in training are copied unchanged
    return "".join(mapping.get(char, char) for char in form)


def rank_sources(train, transform):
    """For each target language, order source languages by average distance on train."""
    distances = defaultdict(list)
    for example in train:
        target_lang = example["target_language"]
        for source_lang, form in example["sources"].items():
            prediction = transform(form, source_lang, target_lang)
            distances[(target_lang, source_lang)].append(edit_distance(prediction, example["target"]))

    ranking = {}
    for target_lang in LANGUAGES:
        scored = [
            (sum(d) / len(d), source_lang)
            for (t, source_lang), d in distances.items()
            if t == target_lang
        ]
        ranking[target_lang] = [source_lang for _, source_lang in sorted(scored)]
    return ranking


def predict(example, ranking, transform):
    target_lang = example["target_language"]
    # Fall back to fixed order if a source language was never seen with this target in training
    order = ranking.get(target_lang, []) + LANGUAGES
    for source_lang in order:
        if source_lang in example["sources"]:
            form = example["sources"][source_lang]
            return transform(form, source_lang, target_lang), source_lang


def evaluate(name, examples, ranking, transform):
    rows = []
    for example in examples:
        prediction, source_lang = predict(example, ranking, transform)
        distance = edit_distance(prediction, example["target"])
        rows.append({
            "baseline": name,
            "target_language": example["target_language"],
            "source_language": source_lang,
            "target": example["target"],
            "prediction": prediction,
            "exact": prediction == example["target"],
            "distance": distance,
            "normalized_distance": distance / len(example["target"]),
        })
    return pd.DataFrame(rows)


def summarize(results):
    grouped = results.groupby("target_language").agg(
        examples=("exact", "size"),
        accuracy=("exact", "mean"),
        edit_distance=("distance", "mean"),
        normalized=("normalized_distance", "mean"),
    )
    grouped.loc["ALL"] = [
        len(results),
        results["exact"].mean(),
        results["distance"].mean(),
        results["normalized_distance"].mean(),
    ]
    grouped["examples"] = grouped["examples"].astype(int)
    return grouped.round(3)


table = pd.read_csv(TABLE_PATH, index_col="cognateset_id")
table["split"] = table.index.map(assign_split)
print("Cognate sets per split:")
print(table["split"].value_counts().to_string())

splits = {name: make_examples(table[table["split"] == name]) for name in ["train", "val", "test"]}
print("\nExamples per split:", {name: len(examples) for name, examples in splits.items()})

mappings = learn_mappings(splits["train"])


def copy_form(form, source_lang, target_lang):
    return form


def map_form(form, source_lang, target_lang):
    return apply_mapping(form, mappings.get((source_lang, target_lang), {}))


baselines = {"copy": copy_form, "mapped": map_form}

for split_name in ["val", "test"]:
    print(f"\n===== {split_name.upper()} =====")
    for name, transform in baselines.items():
        ranking = rank_sources(splits["train"], transform)
        results = evaluate(name, splits[split_name], ranking, transform)
        print(f"\n{name} baseline:")
        print(summarize(results).to_string())

        if split_name == "val" and name == "mapped":
            print("\nSample mapped-baseline errors (val):")
            errors = results[~results["exact"]]
            print(errors[["target_language", "source_language", "target", "prediction"]]
                  .head(15).to_string(index=False))

print("\nLearned correspondences, Hawaiian -> Maori (source:target, '' = deleted):")
hawaiian_maori = mappings.get(("hawaiian", "maori"), {})
print({char: target for char, target in sorted(hawaiian_maori.items()) if char != target})
