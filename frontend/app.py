from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
app = FastAPI()
folder = Path(__file__).resolve().parent
class WordRequest(BaseModel):
    language: str
    concept: str
@app.get("/")
def home():
    return FileResponse(folder / "index.html")
@app.get("/style.css")
def style():
    return FileResponse(folder / "style.css")
@app.post("/predict")
def predict(data: WordRequest):
    return {
        "language": data.language,
        "concept": data.concept,
        "related_words": "Not loaded yet",
        "predicted_word": "Waiting for ML model",
        "actual_word": "Not loaded yet",
        "exact_match": "Not calculated",
        "normalized_edit_distance": "Not calculated"
    }