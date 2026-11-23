"""Unit tests for scripts/metrics.py: the scoring every model is judged by."""
import pandas as pd
import pytest

from metrics import edit_distance, score, summarize


@pytest.mark.parametrize("a, b, expected", [
    ("make", "make", 0),     # identical
    ("mata", "maka", 1),     # one substitution (t -> k)
    ("tama", "tamaa", 1),    # one insertion
    ("aʔu", "au", 1),        # one deletion (glottal stop)
    ("", "ika", 3),          # empty word
    ("fale", "hale", 1),
    ("abc", "xyz", 3),
])
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected


def test_edit_distance_is_symmetric():
    assert edit_distance("whare", "fale") == edit_distance("fale", "whare")


def test_score_exact_match():
    result = score("make", "make")
    assert result["exact"] is True
    assert result["distance"] == 0
    assert result["normalized_distance"] == 0


def test_score_near_miss():
    result = score("make", "mate")          # target, prediction
    assert result["exact"] is False
    assert result["distance"] == 1
    assert result["normalized_distance"] == pytest.approx(0.25)


def test_summarize_adds_overall_row():
    results = pd.DataFrame([
        {"target_language": "hawaiian", "exact": True, "distance": 0, "normalized_distance": 0.0},
        {"target_language": "hawaiian", "exact": False, "distance": 2, "normalized_distance": 0.5},
        {"target_language": "samoan", "exact": True, "distance": 0, "normalized_distance": 0.0},
    ])
    table = summarize(results)
    assert table.loc["hawaiian", "accuracy"] == 0.5
    assert table.loc["ALL", "examples"] == 3
    assert table.loc["ALL", "accuracy"] == pytest.approx(0.667, abs=0.001)
