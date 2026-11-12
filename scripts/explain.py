from rule_explain import explain_sound_rules
from difflib import SequenceMatcher


def explain_prediction(prediction, sources, target_language=None, cognateset_id=None):
    """
    Describe observable similarities between predicted
    words and related-language forms.

    This is a post-hoc comparison, not an explanation
    of the neural network's internal decision process.
    """
    prediction = prediction.strip()
    evidence = []

    for language, form in sources.items():
        form = str(form).strip()

        if not form:
            continue

        matcher = SequenceMatcher(
            None, prediction, form, autojunk=False
        )

        shared = [
            prediction[block.a:block.a + block.size]
            for block in matcher.get_matching_blocks()
            if block.size > 0
        ]

        evidence.append({
            "language": language,
            "form": form,
            "shared_sequences": shared,
            "similarity": round(matcher.ratio(), 3),
            "exact_match": prediction == form
        })

    evidence.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    if not evidence:
        summary = "No related-language forms are available."
    else:
        closest = evidence[0]

        if closest["exact_match"]:
            summary = (
                f"The predicted word '{prediction}' exactly "
                f"matches the {closest['language']} form."
            )
        elif closest["shared_sequences"]:
            shared = ", ".join(
                repr(s) for s in closest["shared_sequences"]
            )
            summary = (
                f"The prediction '{prediction}' shares "
                f"the sequence(s) {shared} with the "
                f"{closest['language']} word '{closest['form']}'."
            )
        else:
            summary = (
                f"The prediction '{prediction}' has no "
                f"contiguous characters in common with "
                f"its closest available source form."
            )

    sound_rules = (
        explain_sound_rules(
            prediction,
            target_language,
            sources,
            cognateset_id
        )
        if target_language else []
    )

    if sound_rules:
        strongest = sound_rules[0]
        summary += (
            f" A recurring candidate correspondence is "
            f"{strongest['target_character']} in "
            f"{target_language} and "
            f"{strongest['source_character']} in "
            f"{strongest['source_language']}, observed in "
            f"{strongest['supporting_sets']} other "
            f"training cognate sets."
        )

    return {
        "sound_correspondences": sound_rules,
        "prediction": prediction,
        "summary": summary,
        "evidence": evidence,
        "disclaimer": (
            "Explanations combine spelling similarities with "
            "candidate sound correspondences observed in "
            "documented training cognates. These patterns "
            "are not independently verified historical sound "
            "laws or direct explanations of the neural "
            "network's internal decisions."
        )
    }
