from difflib import SequenceMatcher
from functools import lru_cache

from correspondences import discover_correspondences
from data import load_examples


@lru_cache(maxsize=1)
def get_rules():
    return discover_correspondences(
        load_examples()["train"],
        min_sets=3
    )


def explain_sound_rules(
    prediction,
    target_language,
    sources,
    cognateset_id=None
):
    target_language = target_language.lower()
    matches = []

    for source_language, source_form in sources.items():
        source_language = source_language.lower()

        if not prediction or not source_form:
            continue

        matcher = SequenceMatcher(
            None, prediction, source_form,
            autojunk=False
        )

        substitutions = set()

        for op, a1, a2, b1, b2 in matcher.get_opcodes():
            if op != "replace":
                continue

            target_char = prediction[a1:a2]
            source_char = source_form[b1:b2]

            if len(target_char) == len(source_char) == 1:
                substitutions.add((target_char, source_char))

        for rule in get_rules():
            if (
                rule["language_a"] == target_language
                and rule["language_b"] == source_language
            ):
                target_char = rule["character_a"]
                source_char = rule["character_b"]

            elif (
                rule["language_b"] == target_language
                and rule["language_a"] == source_language
            ):
                target_char = rule["character_b"]
                source_char = rule["character_a"]

            else:
                continue

            if (target_char, source_char) not in substitutions:
                continue

            supporting_ids = [
                set_id
                for set_id in rule["supporting_set_ids"]
                if str(set_id) != str(cognateset_id)
            ]

            # Require evidence from at least three other sets.
            if len(supporting_ids) < 3:
                continue

            examples = [
                item for item in rule["examples"]
                if str(item["cognateset_id"]) != str(cognateset_id)
            ]

            matches.append({
                "source_language": source_language,
                "source_word": source_form,
                "target_character": target_char,
                "source_character": source_char,
                "supporting_sets": len(supporting_ids),
                "examples": examples[:3]
            })

    matches.sort(
        key=lambda item: item["supporting_sets"],
        reverse=True
    )

    return matches[:5]
