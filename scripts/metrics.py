import pandas as pd

# Shared scoring so the baselines and the model are measured the same way


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


def score(target, prediction):
    """Exact match, edit distance, and edit distance divided by target length."""
    distance = edit_distance(prediction, target)
    return {
        "exact": prediction == target,
        "distance": distance,
        "normalized_distance": distance / len(target),
    }


def summarize(results):
    """Per-language and overall accuracy / edit distance for a results DataFrame."""
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
