import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/abvd/cldf"
OUTPUT = ROOT / "data/processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

LANGUAGES = [
    "hawaiian", "maori", "rarotongan",
    "samoan", "tahitian", "tongan"
]

# Prefer contemporary records. Historical varieties remain in
# the audit output rather than silently becoming modern targets.
LANGUAGE_IDS = {
    52: "hawaiian",
    85: "maori",
    58: "rarotongan",
    118: "samoan",
    173: "tahitian",
    136: "tongan",
    1636: "tahitian",
}

# Explicit preference among contemporary Tahitian sources.
SOURCE_PRIORITY = {173: 0, 1636: 1}

def normalize(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value).strip())
    for char in ["ʻ", "’", "'"]:
        value = value.replace(char, "ʔ")
    value = re.sub(r"[!?()]", "", value)
    return value.strip()



# ABVD's Maori, Rarotongan and Tahitian sources write a long vowel as a doubled
# letter ("taane"), while ACD and ABVD's other languages use a macron ("tāne").
# Rewrite them to macrons so one sound has one spelling across both datasets.
# Samoan and Tongan are left alone: there a double vowel is two syllables
# ("itiiti" = iti + iti), not a long vowel.
DOUBLED_LONG_VOWEL_LANGUAGES = {"maori", "rarotongan", "tahitian"}
LONG_VOWELS = {"aa": "ā", "ee": "ē", "ii": "ī", "oo": "ō", "uu": "ū"}


def long_vowels_to_macrons(form, language):
    if language not in DOUBLED_LONG_VOWEL_LANGUAGES:
        return form
    for doubled, macron in LONG_VOWELS.items():
        form = form.replace(doubled, macron)
    return form

forms = pd.read_csv(SOURCE / "forms.csv", low_memory=False)
cognates = pd.read_csv(SOURCE / "cognates.csv", low_memory=False)
parameters = pd.read_csv(SOURCE / "parameters.csv", low_memory=False)

forms = forms[
    forms["Language_ID"].isin(LANGUAGE_IDS)
    & forms["Loan"].eq(False)
].copy()
forms["language"] = forms["Language_ID"].map(LANGUAGE_IDS)
forms["clean"] = forms["Form"].map(normalize)


# ABVD's Rarotongan source marks a glottal stop with a grave accent on the vowel
# after it ("àre" = ʔare "house", "aì" = aʔi "fire"); ACD writes it as ʔ.
# This must run before the long-vowel step so "àa" becomes ʔaa and then ʔā.
GRAVE_TO_GLOTTAL = {"à": "ʔa", "è": "ʔe", "ì": "ʔi", "ò": "ʔo", "ù": "ʔu"}
is_rarotongan = forms["language"] == "rarotongan"
for grave, glottal in GRAVE_TO_GLOTTAL.items():
    forms.loc[is_rarotongan, "clean"] = forms.loc[is_rarotongan, "clean"].str.replace(grave, glottal, regex=False)

forms["clean"] = [
    long_vowels_to_macrons(form, language)
    for form, language in zip(forms["clean"], forms["language"])
]

# Maori "wh" is one sound written with two letters; write it as "f" to match
# build_cognate_table.py so both datasets spell it the same way.
is_maori = forms["language"] == "maori"
forms.loc[is_maori, "clean"] = forms.loc[is_maori, "clean"].str.replace("wh", "f", regex=False)

# Keep single lexical forms; don't silently concatenate phrases.
forms = forms[
    forms["clean"].ne("")
    & ~forms["clean"].str.contains(r"\s", regex=True)
    & ~forms["clean"].str.startswith("-")
    & ~forms["clean"].str.endswith("-")
].copy()


# Match ACD's spelling of the remaining sounds:
# - "ng" is one sound in every Polynesian language (there is no separate g); ACD writes ŋ.
# - "|" and "-" mark morpheme boundaries inside a word (lo|ko, huru-manu). ACD's
#   builder removes these marks, so do the same here.
# - One Rarotongan form writes the glottal stop as "q" (putaangiqu).
forms["clean"] = (
    forms["clean"]
    .str.replace("ng", "ŋ", regex=False)
    .str.replace("|", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.replace("q", "ʔ", regex=False)
)
# Standard Samoan spelling writes ŋ as "g" (gafulu = ŋafulu "ten").
is_samoan = forms["language"] == "samoan"
forms.loc[is_samoan, "clean"] = forms.loc[is_samoan, "clean"].str.replace("g", "ŋ", regex=False)

joined = forms.merge(
    cognates.loc[
        cognates["Doubt"].eq(False),
        ["Form_ID", "Cognateset_ID"]
    ],
    left_on="ID",
    right_on="Form_ID",
    how="inner",
)

joined = joined.merge(
    parameters[
        ["ID", "Name", "Concepticon_ID", "Concepticon_Gloss"]
    ].rename(columns={
        "ID": "Parameter_ID",
        "Name": "gloss",
    }),
    on="Parameter_ID",
    how="left",
)

# Keep all candidate forms, including conflicts, for inspection.
joined["source_priority"] = joined["Language_ID"].map(
    lambda x: SOURCE_PRIORITY.get(x, 0)
)

joined = joined.sort_values(
    ["Cognateset_ID", "language", "source_priority", "ID"]
)

joined.to_csv(
    OUTPUT / "abvd_variants.csv",
    index=False,
    encoding="utf-8",
)

# For each language/set, prefer the highest-priority source.
# If that source still contains conflicting normalized forms,
# exclude that language/set rather than guessing.
selected_rows = []
ambiguous = []

for (set_id, language), group in joined.groupby(
    ["Cognateset_ID", "language"], sort=True
):
    best_priority = group["source_priority"].min()
    preferred = group[group["source_priority"] == best_priority]
    distinct = preferred["clean"].drop_duplicates()

    if len(distinct) != 1:
        ambiguous.append((set_id, language, len(distinct)))
        continue

    selected_rows.append(preferred.iloc[0])

selected = pd.DataFrame(selected_rows)

# Require evidence from at least two different languages.
coverage = selected.groupby("Cognateset_ID")["language"].nunique()
selected = selected[
    selected["Cognateset_ID"].isin(
        coverage[coverage >= 2].index
    )
].copy()

# Remove exact duplicate sets based on underlying original form
# membership, while retaining the source IDs for traceability.
membership = joined.groupby("Cognateset_ID")["ID"].apply(
    lambda values: tuple(sorted(set(values.astype(str))))
)

seen = {}
canonical = {}

for set_id in sorted(selected["Cognateset_ID"].unique()):
    signature = membership[set_id]
    if signature not in seen:
        seen[signature] = set_id
    canonical[set_id] = seen[signature]

selected["canonical_set"] = selected["Cognateset_ID"].map(canonical)
selected = selected[
    selected["Cognateset_ID"] == selected["canonical_set"]
].copy()

table = selected.pivot(
    index="Cognateset_ID",
    columns="language",
    values="clean",
)

table = table.reindex(columns=LANGUAGES)
table["n_languages"] = table[LANGUAGES].notna().sum(axis=1)

metadata = selected.groupby("Cognateset_ID").first()[
    ["gloss", "Concepticon_ID", "Concepticon_Gloss"]
]

table = table.join(metadata)
table["source"] = "ABVD"
table["proto_language"] = pd.NA
table["proto_form"] = pd.NA
table.index.name = "cognateset_id"

# Connected components: sets sharing a source form ID stay in
# the same evaluation partition, even if their IDs differ.
parent = {s: s for s in selected["Cognateset_ID"].unique()}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

form_to_sets = defaultdict(list)
for row in selected.itertuples():
    form_to_sets[str(row.ID)].append(row.Cognateset_ID)

for ids in form_to_sets.values():
    for other in ids[1:]:
        union(ids[0], other)

table["split_group"] = [
    find(set_id) for set_id in table.index
]

# Deterministic 80/10/10 group assignment.
def split_for_group(group):
    bucket = int(
        hashlib.md5(str(group).encode()).hexdigest(), 16
    ) % 10
    return "val" if bucket == 8 else (
        "test" if bucket == 9 else "train"
    )

table["split"] = table["split_group"].map(split_for_group)

output_path = OUTPUT / "abvd_cognate_table.csv"
table.to_csv(output_path, encoding="utf-8")

print("=== ABVD BUILD COMPLETE ===")
print("Saved:", output_path)
print("Cognate sets:", len(table))
print("Language forms:", int(table["n_languages"].sum()))
print("Ambiguous language/set groups excluded:", len(ambiguous))
print("Variants audit:", OUTPUT / "abvd_variants.csv")
print("\nLanguage coverage:")
print(table[LANGUAGES].notna().sum().to_string())
print("\nSplits:")
print(table["split"].value_counts().to_string())
print("\nSample:")
print(table.head(5).to_string())
