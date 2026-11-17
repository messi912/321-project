
import pandas as pd

df = pd.read_csv("data/processed/cognate_table.csv")

languages = [
    "hawaiian", "maori", "rarotongan",
    "samoan", "tahitian", "tongan"
]

def has_full_reduplication(word):
    """Detect exact repetition of an entire sequence."""
    if not isinstance(word, str):
        return False

    word = word.strip()
    if len(word) < 2 or len(word) % 2 != 0:
        return False

    half = len(word) // 2
    return word[:half] == word[half:]


results = []

for _, row in df.iterrows():
    for language in languages:
        word = row[language]

        if has_full_reduplication(word):
            results.append({
                "cognateset_id": row["cognateset_id"],
                "language": language,
                "word": word,
                "gloss": row["gloss"]
            })

audit = pd.DataFrame(
    results,
    columns=["cognateset_id", "language", "word", "gloss"]
)

print("Full reduplication candidates:", len(audit))
print(audit.head(30).to_string(index=False))

audit.to_csv(
    "data/processed/reduplication_audit.csv",
    index=False
)
