"""Tests for scripts/splits.py: the train/val/test split must be fixed and leak-free."""
import pandas as pd

from conftest import ACD_TABLE, need
from splits import assign_split


def test_split_is_deterministic():
    # The same ID always gets the same split, on every computer
    assert assign_split("24773_4") == assign_split("24773_4")


def test_split_values_are_valid():
    for set_id in ["24773_4", "24790_1", "26754_1", "30115_1"]:
        assert assign_split(set_id) in {"train", "val", "test"}


def test_sets_with_the_same_root_share_a_split():
    # "die" (24773_4) and "kill" (24773_10) come from the same root,
    # so one must never be in training while the other is in testing
    assert assign_split("24773_4") == assign_split("24773_10")


def test_no_root_is_split_across_train_and_test():
    table = pd.read_csv(need(ACD_TABLE), index_col="cognateset_id")
    splits = pd.Series([assign_split(i) for i in table.index], index=table.index)
    roots = table.index.str.split("_").str[0]
    splits_per_root = splits.groupby(roots).nunique()
    assert (splits_per_root == 1).all()


def test_split_sizes_are_roughly_80_10_10():
    table = pd.read_csv(need(ACD_TABLE), index_col="cognateset_id")
    shares = pd.Series([assign_split(i) for i in table.index]).value_counts(normalize=True)
    assert 0.70 < shares["train"] < 0.90
    assert 0.05 < shares["val"] < 0.20
    assert 0.03 < shares["test"] < 0.20
