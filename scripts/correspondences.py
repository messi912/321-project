from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from data import load_examples


def collect_cognate_pairs(train_examples):
    """Collect unique language/form pairs by cognate set."""
    groups = defaultdict(lambda: defaultdict(set))

    for example in train_examples:
        set_id = str(example["cognateset_id"])

        groups[set_id][example["target_language"]].add(
            example["target"]
        )

        for language, form in example["sources"].items():
            if form:
                groups[set_id][language].add(form)

    return groups


def discover_correspondences(train_examples, min_sets=3):
    groups = collect_cognate_pairs(train_examples)
    evidence = defaultdict(set)
    examples = defaultdict(list)

    for set_id, languages in groups.items():
        for lang_a, lang_b in combinations(
            sorted(languages), 2
        ):
            # Skip ambiguous sets with multiple forms per language.
            if len(languages[lang_a]) != 1:
                continue
            if len(languages[lang_b]) != 1:
                continue

            word_a = next(iter(languages[lang_a]))
            word_b = next(iter(languages[lang_b]))

            matcher = SequenceMatcher(
                None, word_a, word_b, autojunk=False
            )

            for operation, a1, a2, b1, b2 in matcher.get_opcodes():
                if operation != "replace":
                    continue

                segment_a = word_a[a1:a2]
                segment_b = word_b[b1:b2]

                # First version: only unambiguous single-character
                # substitutions. Exclude insertions and deletions.
                if len(segment_a) != 1 or len(segment_b) != 1:
                    continue

                key = (
                    lang_a, lang_b,
                    segment_a, segment_b
                )

                evidence[key].add(set_id)

                if len(examples[key]) < 5:
                    examples[key].append({
                        "cognateset_id": set_id,
                        "word_a": word_a,
                        "word_b": word_b
                    })

    results = []

    for key, supporting_sets in evidence.items():
        if len(supporting_sets) < min_sets:
            continue

        lang_a, lang_b, char_a, char_b = key

        results.append({
            "language_a": lang_a,
            "language_b": lang_b,
            "character_a": char_a,
            "character_b": char_b,
            "supporting_sets": len(supporting_sets),
            "supporting_set_ids": sorted(supporting_sets),
            "examples": examples[key]
        })

    results.sort(
        key=lambda item: item["supporting_sets"],
        reverse=True
    )

    return results


if __name__ == "__main__":
    splits = load_examples()

    rules = discover_correspondences(
        splits["train"],
        min_sets=3
    )

    print("=== CANDIDATE SOUND CORRESPONDENCES ===")
    print("Training examples:", len(splits["train"]))
    print("Candidate rules:", len(rules))

    for rule in rules[:20]:
        print()
        print(
            f"{rule['language_a']} "
            f"{rule['character_a']} ↔ "
            f"{rule['language_b']} "
            f"{rule['character_b']}"
        )
        print(
            "Supporting cognate sets:",
            rule["supporting_sets"]
        )

        for example in rule["examples"][:3]:
            print(
                f"  {example['word_a']} ↔ "
                f"{example['word_b']} "
                f"[{example['cognateset_id']}]"
            )
