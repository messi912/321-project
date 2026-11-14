"""Combine ACD and ABVD into one training table.

ACD keeps its usual root-based split. ABVD sets are added as extra training data,
except any set that shares a word with an ACD val/test set, so test words never leak.
"""
from pathlib import Path

import pandas as pd

from splits import assign_split

ROOT = Path(__file__).resolve().parents[1]
ACD_PATH = ROOT / "data" / "processed" / "cognate_table.csv"
ABVD_PATH = ROOT / "data" / "processed" / "abvd_cognate_table.csv"
OUT_PATH = ROOT / "data" / "processed" / "combined_table.csv"

LANGUAGES = ["hawaiian", "maori", "rarotongan", "samoan", "tahitian", "tongan"]

acd = pd.read_csv(ACD_PATH, index_col="cognateset_id")
acd["split"] = [assign_split(set_id) for set_id in acd.index]
abvd = pd.read_csv(ABVD_PATH, index_col="cognateset_id")

# Every word that appears in an ACD val/test set, as (language, word)
held_out = acd[acd["split"] != "train"]
held_out_words = {(lang, word) for lang in LANGUAGES for word in held_out[lang].dropna()}


def shares_held_out_word(row):
    return any(
        (lang, row[lang]) in held_out_words
        for lang in LANGUAGES
        if isinstance(row[lang], str)
    )


extra = abvd[~abvd.apply(shares_held_out_word, axis=1)].copy()
extra["split"] = "train"
extra.index = "abvd:" + extra.index.astype(str)

columns = LANGUAGES + ["gloss", "split"]
combined = pd.concat([acd[columns], extra.reindex(columns=columns)])
combined.index.name = "cognateset_id"
combined.to_csv(OUT_PATH, encoding="utf-8")

print(f"ACD sets: {len(acd)} ({(acd['split'] == 'train').sum()} train)")
print(f"ABVD sets added to training: {len(extra)} (dropped {len(abvd) - len(extra)} that share a test word)")
print(f"Wrote {len(combined)} sets to {OUT_PATH.relative_to(ROOT)}")