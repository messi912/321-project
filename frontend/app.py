from pathlib import Path
import sys
from collections import Counter
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from data import LANGUAGES, TABLE_PATH, load_examples
from train import load_checkpoint
from metrics import score
from explain import explain_prediction
from ensemble import load_models, predict_top

app = FastAPI()


class WordRequest(BaseModel):
    language: str
    concept: str


@lru_cache(maxsize=1)
def get_resources():
    if not TABLE_PATH.exists():
        raise RuntimeError(
            "Missing cognate table. Run scripts/build_cognate_table.py."
        )

    table = pd.read_csv(TABLE_PATH, index_col="cognateset_id")
    examples = load_examples()

    # Use every seq2seq_<seed>.pt as an ensemble; fall back to the
    # single seq2seq.pt if no seed checkpoints have been trained.
    try:
        models, vocab = load_models()
    except FileNotFoundError:
        model, vocab = load_checkpoint()
        models = [model]

    # Match each example to its documented gloss.
    choices = []
    for split_name, split_examples in examples.items():
        for example in split_examples:
            set_id = example["cognateset_id"]
            gloss = str(table.loc[set_id, "gloss"]).strip()
            if not gloss or gloss.lower() == "nan":
                continue

            choices.append({
                **example,
                "gloss": gloss,
                "split": split_name,
            })

    return models, vocab, choices


@app.get("/")
def home():
    return FileResponse(FOLDER / "index.html")


@app.get("/about")
def about():
    return FileResponse(FOLDER / "about.html")


@app.get("/style.css")
def style():
    return FileResponse(FOLDER / "style.css")


@app.get("/concepts")
def concepts(language: str):
    target = language.strip().lower()
    if target not in LANGUAGES:
        raise HTTPException(400, "Unsupported language")

    _, _, choices = get_resources()

    # Multiple sets can share a gloss, so the sister words tell them apart.
    concepts = []
    for item in choices:
        if item["target_language"] != target:
            continue
        sisters = ", ".join(
            f"{lang.title()} {form}" for lang, form in item["sources"].items()
        )
        concepts.append({
            "id": str(item["cognateset_id"]),
            "gloss": item["gloss"],
            "label": f'{item["gloss"]} ({sisters})',
        })

    # The page matches the chosen label exactly, so fall back to the set ID
    # if two sets still end up with the same label.
    counts = Counter(concept["label"] for concept in concepts)
    for concept in concepts:
        if counts[concept["label"]] > 1:
            concept["label"] += f' [{concept["id"]}]'
    return concepts


@app.post("/predict")
def predict_word(data: WordRequest):
    target = data.language.strip().lower()
    if target not in LANGUAGES:
        raise HTTPException(400, "Unsupported language")

    models, vocab, choices = get_resources()

    # The frontend sends a cognate-set ID, not an arbitrary word.
    matches = [
        item for item in choices
        if item["target_language"] == target
        and str(item["cognateset_id"]) == data.concept.strip()
    ]

    if not matches:
        raise HTTPException(
            404,
            "No documented cognate example for this selection."
        )

    example = matches[0]

    # The models receive the target-language tag and sister
    # forms, but NOT the actual target word.
    # Beam search returns ranked (word, confidence) pairs; the top one is the prediction.
    top = predict_top(models, vocab, example)
    prediction = top[0][0]
    actual = example["target"]
    metrics = score(actual, prediction)

    return {
        "language": data.language,
        "concept": example["gloss"],
        "cognateset_id": str(example["cognateset_id"]),
        "split": example["split"],
        "related_words": example["sources"],
        "explanation": explain_prediction(
            prediction,
            example["sources"],
            target_language=example["target_language"],
            cognateset_id=example["cognateset_id"]
        ),
        "predicted_word": prediction,
        # Confidence is the average per-letter probability, so the
        # alternatives' scores do not add up to 1.
        "alternatives": [
            {"word": word, "confidence": confidence, "is_actual": word == actual}
            for word, confidence in top
        ],
        "model_count": len(models),
        "actual_word": actual,
        "exact_match": metrics["exact"],
        "normalized_edit_distance": round(
            metrics["normalized_distance"], 3
        ),
    }