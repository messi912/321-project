from collections import Counter
from difflib import SequenceMatcher

from rule_explain import explain_sound_rules


def readable_languages(languages):
    names = [name.capitalize() for name in languages]

    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    return ", ".join(names[:-1]) + f", and {names[-1]}"


def explain_prediction(
    prediction,
    sources,
    target_language=None,
    cognateset_id=None
):
    """
    Summarize observable linguistic evidence for a prediction.

    This is a post-hoc analysis, not an interpretation
    of the neural network's internal computations.
    """
    prediction = str(prediction).strip()
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

    sound_rules = (
        explain_sound_rules(
            prediction,
            target_language,
            sources,
            cognateset_id
        )
        if target_language else []
    )

    # Identify how many related languages use the same form.
    form_counts = Counter(
        item["form"] for item in evidence
    )

    agreeing = [
        item["language"]
        for item in evidence
        if item["form"] == prediction
    ]

    if not evidence:
        summary = (
            "No related-language forms are available "
            "to support this prediction."
        )

    elif len(agreeing) >= 2:
        languages = readable_languages(agreeing)
        total = len(evidence)

        if len(agreeing) == total:
            summary = (
                f"All {total} related languages "
                f"({languages}) use '{prediction}'. "
                f"Their agreement provides consistent "
                f"cross-language evidence for the predicted "
                f"form, with no spelling differences to reconcile."
            )
        else:
            summary = (
                f"{len(agreeing)} of {total} related languages "
                f"({languages}) use '{prediction}'. "
                f"This agreement supports the predicted form, "
                f"although other related languages differ."
            )

    elif sound_rules:
        strongest = sound_rules[0]

        target_name = (
            target_language.capitalize()
            if target_language else "target"
        )
        source_name = strongest["source_language"].capitalize()

        summary = (
            f"The predicted {target_name} form '{prediction}' "
            f"is consistent with a recurring character "
            f"correspondence: '{strongest['target_character']}' "
            f"in {target_name} aligns with "
            f"'{strongest['source_character']}' in {source_name}. "
            f"This pattern was observed in "
            f"{strongest['supporting_sets']} other "
            f"training cognate sets."
        )

        examples = strongest.get("examples", [])

        if examples:
            example_pairs = [
                f"{item['word_a']}/{item['word_b']}"
                for item in examples[:3]
            ]

            summary += (
                " Supporting examples include "
                + ", ".join(example_pairs)
                + "."
            )

        if agreeing:
            summary += (
                f" The predicted form also exactly matches "
                f"the {readable_languages(agreeing)} word."
            )

    elif agreeing:
        summary = (
            f"The predicted form '{prediction}' matches "
            f"the {readable_languages(agreeing)} word. "
            f"Other related forms provide additional context, "
            f"but no supported character substitution "
            f"was identified for this example."
        )

    else:
        closest = evidence[0]

        if closest["shared_sequences"]:
            summary = (
                f"The prediction '{prediction}' shares "
                f"some spelling features with the "
                f"{closest['language'].capitalize()} word "
                f"'{closest['form']}', but no sufficiently "
                f"supported character correspondence "
                f"was identified."
            )
        else:
            summary = (
                f"The prediction '{prediction}' does not "
                f"share a contiguous character sequence "
                f"with its closest related form. "
                f"The available linguistic evidence "
                f"is limited."
            )

    return {
        "prediction": prediction,
        "summary": summary,
        "evidence": evidence,
        "sound_correspondences": sound_rules,
        "disclaimer": (
            "Evidence-based comparison; not a direct "
            "explanation of the model's internal reasoning."
        )
    }
