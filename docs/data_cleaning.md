# Data Cleaning Rules

**Owner:** CJ, Jyoti, Misgana
**Script:** `scripts/build_cognate_table.py` (written by CJ)
**Input:** ACD CLDF data in `data/acd/cldf/` (`forms.csv`, `cognates.csv`, `cognatesets.csv`), ACD v2.0
**Output:** `data/processed/cognate_table.csv`, one row per cognate set, one column per language, used by (`baseline.py`, `data.py`)

## Goal
The raw ACD data is in CSV form with one row per word, with messy spellings, loans, and several forms per language. The model needs the data reshaped into one row per word family, with one comparable form per language.

## Pipeline at a glance

| Step | Rule | Rows remaining |
|---|---|---|
| 0 | Forms from the 6 candidate languages | 2,981 forms |
| 1 | Remove loanwords | 2,960 |
| 2 | Keep only forms that belong to a cognate set | 2,892 |
| 3 | Normalize spelling | 2,892 |
| 4 | Remove affixes and phrases | 2,704 |
| 5 | Keep one form per language per cognate set | 2,150 |
| 6 | Pivot to one row per cognate set; keep sets with ≥ 2 languages | **562 sets** |

## Rules

### 1. Remove loanwords
**What:** Drop forms where `Loan = true` (21 forms).

**Why:** Loanwords do not follow sound changes in regular patterns the model is training on.

**Code:** line 56

### 2. Keep only forms with a cognate set
**What:** Join `forms.csv` to `cognates.csv` on `ID = Form_ID`; drop forms with no match (68 forms).

**Why:** Cognate sets are the only way the model compares words with sister languages.

**Code:** lines 59–64

### 3. Normalize spelling
**What:**
- Unicode NFC normalization (so "ā" is always stored the same way)
- Variants: `hae ~ hae-hae` → keep first (`hae`)
- Glottal stop: `ʻ`, `'`, `’`, `Ɂ` → `ʔ`
- Remove `( ) ! ?`

**Why:** The model may see these variants as sound changing patterns but it needs to see them as the same.

**Code:** `normalize()`, lines 27–35

### 4. Remove affixes and phrases
**What:** Drop forms starting or ending with `-` (affixes like `-hia`) and forms containing a space (phrases like `ake māmā`). 188 forms removed.

**Why:** The model will count affixes and phrases as single words and mess up the training by comparing it with one sister language word.

**Code:** lines 70–72

### 5. One form per language per cognate set
**What:** When a language has several forms in one set, keep the one with the fewest hyphens, then the shortest. Then remove hyphens.
Example, "die" (`24773_4`): Samoan has `mate` and `ta-mate` → keep `mate`.

**Why:** The shortest form removes any prefixes or suffixes and is usually closer to the word's root.

**Code:** lines 77–81

### 6. Pivot and minimum coverage
**What:** Reshape to one row per cognate set with a column per language; drop sets with fewer than 2 languages. Attach the proto-form and gloss from `cognatesets.csv`.

**Why:** Each example hides one language and predicts it from the remaining ones, so a set needs at least two languages: one to hide and at least one to predict from.

**Code:** lines 85–93

## Known issues
- **Tahitian:** only 2 forms survive cleaning; effectively unusable.
- **Multi-letter sounds:** Māori `wh` is one sound written as two letters, so the baseline treats it as two. Example error: target `hoʔokani`, predicted `whaʔakani`.
- **Reduplication mismatches:** one language's form is doubled and another's isn't (`pulu` vs `pulupulu`); no model can predict these correctly.
- **Split leakage:** 8 of 61 test sets share a root with a training set (e.g. "die" `make` and "kill" `hoʔomake`, both root 24773).

## Proposed changes
### 1. Split by root instead of by cognate set
**Problem:** 8 of 61 test sets share a root with a training set, so the model gets a small preview of some test words (e.g. "die" `make` and "kill" `hoʔomake`).
**Change:** Hash only the root part of the ID (`24773` from `24773_4`) so all sets
from one root land in the same split.
**Where:** `assign_split()` in `splits.py` (one line).
**Measure:** Test sets sharing a root with training should drop from 8 to 0.
Accuracy may drop slightly — that is expected, because the current number is slightly inflated by the leak. The new number is the honest one.

### 2. Handle reduplication mismatches
**Problem:** Some languages list a doubled form (`pulupulu`) where sisters list the single form (`pulu`), so no model can predict them correctly.
**Change:** When a form is an exact doubling (first half = second half) and at least one sister has the single half, keep the single half.
**Where:** Add a step after Rule 5 in `build_cognate_table.py`.
**Measure:** Count how many forms are changed, then rerun `baseline.py`.
Accuracy should improve slightly, since these were guaranteed errors.