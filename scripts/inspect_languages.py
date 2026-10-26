import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "acd" / "cldf"

# Load the language information from the CSV file
languages = pd.read_csv(DATA_DIR / "languages.csv")

# Count how many language records are in the dataset and search for initial Polynesian language candidates
print("Total languages:", len(languages))

polynesian = languages[
    languages["Name"].str.contains(
        "Maori|Māori|Hawaiian|Samoan|Tongan|Tahitian",
        case=False,
        na=False
    )
]

print(polynesian[["ID", "Name", "Group"]].to_string(index=False))

# Load the recorded word forms
forms = pd.read_csv(DATA_DIR / "forms.csv", low_memory=False)

# Keep only forms belonging to candidate languages
polynesian_forms = forms[
    forms["Language_ID"].isin(polynesian["ID"])
]

form_counts = polynesian_forms.groupby("Language_ID").size()

print("\nWord forms per language:")
print(form_counts)


# Count how many different languages have a form and for each parameter (potential shared concept)
concept_overlap = (
    polynesian_forms.groupby("Parameter_ID")["Language_ID"]
    .nunique()
)

print("\nConcept overlap:")
print("At least 2 languages:", (concept_overlap >= 2).sum())
print("At least 3 languages:", (concept_overlap >= 3).sum())
print("At least 4 languages:", (concept_overlap >= 4).sum())
print("At least 5 languages:", (concept_overlap >= 5).sum())

# Load the meanings dataset
parameters = pd.read_csv(DATA_DIR / "parameters.csv")

# Connect each Polynesian word to its meaning
words_with_meanings = polynesian_forms.merge(
    parameters[["ID", "Name"]],
    left_on="Parameter_ID",
    right_on="ID",
    how="left"
)

# Display a small sample
print("\nPolynesian words with meanings:")
print(
    words_with_meanings[
        ["Language_ID", "Form", "Name"]
    ].head(15).to_string(index=False)
)

# Load cognate assignments
cognates = pd.read_csv(DATA_DIR / "cognates.csv", low_memory=False)

# Connect Polynesian word forms to their cognate sets
polynesian_cognates = polynesian_forms.merge(
    cognates[["Form_ID", "Cognateset_ID"]],
    left_on="ID",
    right_on="Form_ID",
    how="inner"
)

# Count how many distinct languages appear in each cognate set
cognate_overlap = (
    polynesian_cognates.groupby("Cognateset_ID")["Language_ID"]
    .nunique()
)

print("\nCognate set overlap:")
print("At least 2 languages:", (cognate_overlap >= 2).sum())
print("At least 3 languages:", (cognate_overlap >= 3).sum())
print("At least 4 languages:", (cognate_overlap >= 4).sum())

# Display a few examples from shared cognate sets
shared_sets = cognate_overlap[cognate_overlap >= 2].index

examples = polynesian_cognates[
    polynesian_cognates["Cognateset_ID"].isin(shared_sets)
]

print("\nExamples of shared cognates:")
print(
    examples[
        ["Language_ID", "Form", "Cognateset_ID"]
    ].head(20).to_string(index=False)
)