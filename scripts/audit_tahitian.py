
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLDF = ROOT / "data/acd/cldf"

languages = pd.read_csv(CLDF / "languages.csv")
forms = pd.read_csv(CLDF / "forms.csv")
cognates = pd.read_csv(CLDF / "cognates.csv")

print("LANGUAGE COLUMNS:", languages.columns.tolist())
print("FORM COLUMNS:", forms.columns.tolist())
print("COGNATE COLUMNS:", cognates.columns.tolist())

# Identify Tahitian in the source language table
matches = languages[
    languages.astype(str)
    .apply(lambda col: col.str.contains("tahitian", case=False, na=False))
    .any(axis=1)
]

print("\nTAHITIAN LANGUAGE RECORDS")
print(matches.to_string(index=False))

# Find raw forms for the matching language IDs
if "ID" in languages.columns and "Language_ID" in forms.columns:
    ids = set(matches["ID"].astype(str))
    tahitian = forms[
        forms["Language_ID"].astype(str).isin(ids)
    ].copy()

    print("\nTotal raw Tahitian forms:", len(tahitian))
    print(tahitian.head(20).to_string(index=False))

    print("\nRaw Tahitian forms with cognate annotations:")
    if "Form_ID" in cognates.columns and "ID" in forms.columns:
        linked = tahitian[
            tahitian["ID"].astype(str).isin(
                cognates["Form_ID"].astype(str)
            )
        ]
        print(len(linked))
    else:
        print("Check cognate column names above.")
