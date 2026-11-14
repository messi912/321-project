import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "acd" / "cldf"
OUT_DIR = ROOT / "data" / "processed"

# ACD language IDs for the six candidate Polynesian languages
LANGUAGES = {
    620: "hawaiian",
    330: "maori",
    388: "rarotongan",
    401: "samoan",
    1040: "tahitian",
    434: "tongan",
}

MIN_LANGUAGES = 2

# Sources write the glottal stop several different ways
GLOTTAL_STOPS = {"Ɂ": "ʔ", "ʻ": "ʔ", "'": "ʔ", "’": "ʔ"}


def normalize(form):
    """Clean a raw ACD form, keeping morpheme hyphens so we can rank forms later."""
    form = unicodedata.normalize("NFC", str(form)).strip()
    # "matua ~ metua" lists alternatives; keep the first one
    form = form.split("~")[0].strip()
    for variant, standard in GLOTTAL_STOPS.items():
        form = form.replace(variant, "ʔ")
    form = re.sub(r"[()!?]", "", form)
    return form


def parse_etymon(name):
    """Split a cognate set name like "POC *koro₂ 'fortified place'" into its parts."""
    match = re.match(r"^(\S+) \*(\S+) '(.*)'$", str(name))
    if not match:
        return pd.Series([None, None, None])
    proto_language, proto_form, gloss = match.groups()
    # Subscript digits only distinguish homophones in the ACD
    proto_form = re.sub(r"[₀-₉]", "", proto_form)
    return pd.Series([proto_language, proto_form, gloss])


def single_half(form):
    """Return the half of an exactly doubled form ("piapia" -> "pia"), or None."""
    half = len(form) // 2
    if len(form) >= 4 and len(form) % 2 == 0 and form[:half] == form[half:]:
        return form[:half]
    return None


# Load the CLDF tables
forms = pd.read_csv(DATA_DIR / "forms.csv", low_memory=False)
cognates = pd.read_csv(DATA_DIR / "cognates.csv", low_memory=False)
cognatesets = pd.read_csv(DATA_DIR / "cognatesets.csv")

# Keep forms from the candidate languages, excluding known loanwords
forms = forms[forms["Language_ID"].isin(LANGUAGES)]
forms = forms[forms["Loan"] != True]
print("Candidate forms (no loans):", len(forms))

forms = forms.merge(
    cognates[["Form_ID", "Cognateset_ID"]],
    left_on="ID",
    right_on="Form_ID",
    how="inner",
)
print("Forms with a cognate set:", len(forms))

forms["Clean"] = forms["Form"].map(normalize)


# Maori "wh" is one sound (close to f) written with two letters. The models read one
# letter at a time, so "wh" was learned as w + h and mapped to "vf" in other languages.
# Write it as "f", which also matches its Samoan and Tongan cognates (whetū = fetū).
is_maori = forms["Language_ID"] == 330
forms.loc[is_maori, "Clean"] = forms.loc[is_maori, "Clean"].str.replace("wh", "f", regex=False)


# A few ACD entries use letters these languages don't have (Rarotongan "wai" for
# vai "water", "lanu" for ranu). Rarotongan has v and r, Tongan has v.
LETTER_FIXES = {388: {"w": "v", "l": "r"}, 434: {"w": "v"}}
for language_id, fixes in LETTER_FIXES.items():
    is_language = forms["Language_ID"] == language_id
    for wrong, right in fixes.items():
        forms.loc[is_language, "Clean"] = forms.loc[is_language, "Clean"].str.replace(wrong, right, regex=False)

# Drop bound affixes like "-a" or "faʔa-" and multi-word phrases
is_affix = forms["Clean"].str.startswith("-") | forms["Clean"].str.endswith("-")
is_phrase = forms["Clean"].str.contains(" ")
forms = forms[~is_affix & ~is_phrase & (forms["Clean"] != "")]
print("After dropping affixes and phrases:", len(forms))

# A language often has several forms in one cognate set (e.g. "mate" and "ta-mate").
# Prefer the root: fewest morpheme boundaries, then shortest.
forms["Boundaries"] = forms["Clean"].str.count("-")
forms["Length"] = forms["Clean"].str.len()
forms = forms.sort_values(["Cognateset_ID", "Language_ID", "Boundaries", "Length"])
best = forms.drop_duplicates(["Cognateset_ID", "Language_ID"], keep="first").copy()
best["Clean"] = best["Clean"].str.replace("-", "", regex=False)
print("Forms after picking one per language per set:", len(best))

# Some sources list a doubled form (Hawaiian "piapia") where a sister in the same cognate
# set lists the single form ("pia"). No model can predict that difference, so in that
# case keep the single half. A doubled form with no single-form sister is left alone.
forms_in_set = best.groupby("Cognateset_ID")["Clean"].agg(set)


def collapse(row):
    half = single_half(row["Clean"])
    if half is not None and half in forms_in_set[row["Cognateset_ID"]]:
        return half
    return row["Clean"]


collapsed = best.apply(collapse, axis=1)
print("Reduplicated forms collapsed:", (collapsed != best["Clean"]).sum())
best["Clean"] = collapsed

# One row per cognate set, one column per language
table = best.pivot(index="Cognateset_ID", columns="Language_ID", values="Clean")
table = table.rename(columns=LANGUAGES)[list(LANGUAGES.values())]
table["n_languages"] = table.notna().sum(axis=1)
table = table[table["n_languages"] >= MIN_LANGUAGES]

# Attach the ancestor (proto) form and meaning recorded for each set
etyma = cognatesets.set_index("ID")["Name"].apply(parse_etymon)
etyma.columns = ["proto_language", "proto_form", "gloss"]
table = table.join(etyma, how="left")
table.index.name = "cognateset_id"

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "cognate_table.csv"
table.to_csv(out_path, encoding="utf-8")

print(f"\nWrote {len(table)} cognate sets to {out_path.relative_to(ROOT)}")
print("\nSets by number of languages:")
print(table["n_languages"].value_counts().sort_index().to_string())
print("\nForms per language:")
print(table[list(LANGUAGES.values())].notna().sum().to_string())
print("\nCharacters used:")
characters = "".join(table[list(LANGUAGES.values())].stack().dropna())
print("".join(sorted(set(characters))))
print("\nSample:")
print(table.head(10).to_string())