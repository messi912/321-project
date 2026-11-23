"""Shared setup for all tests.

Lets the tests import the project's scripts, and gives paths to the built data
and trained models. Tests that need data or models skip themselves (instead of
failing) when those files haven't been built yet.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "frontend"))

PROCESSED = ROOT / "data" / "processed"
ACD_TABLE = PROCESSED / "cognate_table.csv"
ABVD_TABLE = PROCESSED / "abvd_cognate_table.csv"
COMBINED_TABLE = PROCESSED / "combined_table.csv"
MODEL_DIR = ROOT / "data" / "models"

LANGUAGES = ["hawaiian", "maori", "rarotongan", "samoan", "tahitian", "tongan"]


def need(path):
    """Skip the test if a built file is missing."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not built yet")
    return path


def need_models():
    if not list(MODEL_DIR.glob("seq2seq_*.pt")):
        pytest.skip("no trained seq2seq_*.pt models yet")
