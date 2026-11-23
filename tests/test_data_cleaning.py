"""Data validation tests for the cleaned tables.

Each test checks one cleaning rule from docs/data_cleaning.md against the tables
the build scripts produce, so a change that breaks a rule is caught right away.
"""
import re

import pandas as pd

from conftest import ABVD_TABLE, ACD_TABLE, COMBINED_TABLE, LANGUAGES, need
from splits import assign_split

# Every sound the cleaned data is allowed to use (ACD's alphabet)
ALLOWED = set("aefhiklmnoprstuvwāēīōūŋʔ")


def words(table, languages=LANGUAGES):
    """All (language, word) pairs in a table."""
    return [(lang, w) for lang in languages if lang in table for w in table[lang].dropna()]


def load(path):
    return pd.read_csv(need(path), index_col="cognateset_id")


def test_acd_table_has_expected_size():
    table = load(ACD_TABLE)
    assert len(table) == 562
    assert table.index.is_unique


def test_every_set_has_at_least_two_languages():
    # Rule 6: one language to hide and at least one to predict from
    table = load(ACD_TABLE)
    assert (table[LANGUAGES].notna().sum(axis=1) >= 2).all()


def test_no_affixes_phrases_or_boundary_marks():
    # Rule 4 and 5: no "-hia", no "ake māmā", no "lo|ko"
    for path in [ACD_TABLE, ABVD_TABLE]:
        for lang, word in words(load(path)):
            assert not re.search(r"[\s\-|~()?!]", word), f"{lang} {word!r} in {path.name}"


def test_only_known_sounds_are_used():
    # Rule 3 and the ABVD spelling fixes: one spelling per sound across both datasets
    for path in [ACD_TABLE, ABVD_TABLE]:
        for lang, word in words(load(path)):
            extra = set(word) - ALLOWED
            assert not extra, f"{lang} {word!r} uses {extra} in {path.name}"


def test_maori_wh_is_written_as_f():
    for path in [ACD_TABLE, ABVD_TABLE]:
        assert not any("wh" in w for w in load(path)["maori"].dropna())


def test_abvd_long_vowels_use_macrons():
    # ABVD wrote "taane"; it must now be "tāne" like ACD
    table = load(ABVD_TABLE)
    for lang in ["maori", "rarotongan", "tahitian"]:
        doubled = [w for w in table[lang].dropna() if re.search(r"(aa|ee|ii|oo|uu)", w)]
        assert not doubled, f"{lang}: {doubled[:5]}"


def test_rarotongan_and_tongan_letter_fixes():
    table = load(ACD_TABLE)
    assert not any(c in w for w in table["rarotongan"].dropna() for c in "lw")
    assert not any("w" in w for w in table["tongan"].dropna())


def test_doubled_words_are_collapsed_when_a_sister_has_the_half():
    # The reduplication rule: no "piapia" in a set where a sister says "pia"
    table = load(ACD_TABLE)
    for set_id, row in table.iterrows():
        forms = {lang: row[lang] for lang in LANGUAGES if isinstance(row[lang], str)}
        for lang, word in forms.items():
            half = len(word) // 2
            if len(word) >= 4 and len(word) % 2 == 0 and word[:half] == word[half:]:
                assert word[:half] not in forms.values(), f"{set_id} {lang} {word}"


def test_combined_table_never_leaks_test_words_into_training():
    # The leakage guard in build_combined_table.py
    combined = load(COMBINED_TABLE)
    acd = combined[~combined.index.str.startswith("abvd:")]
    abvd = combined[combined.index.str.startswith("abvd:")]
    assert (abvd["split"] == "train").all()

    held_out = acd[acd["split"] != "train"]
    held_out_words = set(words(held_out))
    leaked = [pair for pair in words(abvd) if pair in held_out_words]
    assert not leaked, leaked[:5]


def test_combined_table_keeps_the_acd_split():
    combined = load(COMBINED_TABLE)
    acd = combined[~combined.index.str.startswith("abvd:")]
    assert all(acd.loc[i, "split"] == assign_split(i) for i in acd.index)
