# Polynesian Language Reconstruction

CS 321 team project exploring word reconstruction using related Polynesian languages.

## Current progress
- Inspected ACD using Python and Pandas.
- Identified six candidate Polynesian languages.
- Found 2,981 recorded word forms across those candidates.
- Found 579 cognate sets shared by at least two candidate languages.
- Investigating ABVD as an additional dataset.

## Dataset
ACD: https://github.com/lexibank/acd

## Status
Initial dataset exploration completed. Data cleaning, model training, and backend development are still in progress.

## For Use

### Setup (once per computer)
```powershell
# Download the ACD dataset into data/acd (data/ is git-ignored, so everyone downloads it themselves).
# v2.0 is the version our numbers come from.
git clone --branch v2.0 https://github.com/lexibank/acd data/acd

# Install Python packages (CPU-only PyTorch is enough)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Running the scripts
Run these from the project root, in this order:

```powershell
python -X utf8 scripts/build_cognate_table.py   # 1. build the cleaned table
python -X utf8 scripts/baseline.py              # 2. score the baselines
python -X utf8 scripts/data.py                  # 3. check the model's data encoding
python -X utf8 scripts/train.py                 # 4. train the model (~5 min on CPU), saves data/models/seq2seq.pt
python -X utf8 scripts/evaluate.py              # 5. score the saved model on val (add "test" for final numbers)
```
