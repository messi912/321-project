
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/processed/cognate_table.csv"

LANGUAGES = [
    "hawaiian",
    "maori",
    "rarotongan",
    "samoan",
    "tahitian",
    "tongan",
]

df = pd.read_csv(TABLE)

# Number of languages represented in each cognate set
df["language_count"] = df[LANGUAGES].notna().sum(axis=1)

print("\nCOGNATE DATASET COVERAGE")
print("=" * 45)

print("Total cognate sets:", len(df))
print("Total documented forms:", df[LANGUAGES].notna().sum().sum())

print("\nForms per language:")
for lang in LANGUAGES:
    print(f"  {lang}: {df[lang].notna().sum()}")

print("\nCognate sets by language coverage:")
for count, frequency in df["language_count"].value_counts().sort_index().items():
    print(f"  {count} languages: {frequency} sets")

print("\nPairwise language overlap:")
overlap = pd.DataFrame(index=LANGUAGES, columns=LANGUAGES)

for lang1 in LANGUAGES:
    for lang2 in LANGUAGES:
        overlap.loc[lang1, lang2] = int(
            (df[lang1].notna() & df[lang2].notna()).sum()
        )

print(overlap.to_string())

# Save a report of sparse cognate sets
sparse = df[df["language_count"] <= 2].copy()

OUTPUT = ROOT / "data/processed/sparse_cognate_sets.csv"
sparse.to_csv(OUTPUT, index=False)

print(f"\nSparse sets (1-2 languages): {len(sparse)}")
print(f"Saved report to: {OUTPUT}")
