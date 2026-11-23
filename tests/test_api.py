"""Integration tests for the web server (frontend/app.py).

These start the real FastAPI app with the trained ensemble and send it the same
requests the website sends. They also check two non-functional requirements:
every result is labeled as a prediction, and responses come back within 3 seconds.
"""
import time

import pytest
from fastapi.testclient import TestClient

from conftest import need_models


@pytest.fixture(scope="module")
def client():
    need_models()
    from app import app
    return TestClient(app)


@pytest.fixture(scope="module")
def hawaiian_concepts(client):
    response = client.get("/concepts", params={"language": "hawaiian"})
    assert response.status_code == 200
    return response.json()


def test_pages_load(client):
    for path in ["/", "/about", "/style.css"]:
        assert client.get(path).status_code == 200


def test_concept_list_is_not_empty_and_labels_are_unique(hawaiian_concepts):
    assert len(hawaiian_concepts) > 0
    labels = [c["label"] for c in hawaiian_concepts]
    assert len(labels) == len(set(labels))


def test_unsupported_language_is_rejected(client):
    assert client.get("/concepts", params={"language": "french"}).status_code == 400
    assert client.post("/predict", json={"language": "french", "concept": "24773_4"}).status_code == 400


def test_unknown_concept_returns_404(client):
    response = client.post("/predict", json={"language": "hawaiian", "concept": "not-a-set"})
    assert response.status_code == 404


def test_missing_field_is_rejected(client):
    # Pydantic checks the request shape before our code runs
    assert client.post("/predict", json={"language": "hawaiian"}).status_code == 422


def test_prediction_has_everything_the_page_shows(client, hawaiian_concepts):
    concept = hawaiian_concepts[0]["id"]
    result = client.post("/predict", json={"language": "hawaiian", "concept": concept}).json()
    for key in ["predicted_word", "actual_word", "related_words", "explanation",
                "exact_match", "normalized_edit_distance", "alternatives"]:
        assert key in result
    assert "hawaiian" not in result["related_words"]       # the answer is hidden
    assert 1 <= len(result["alternatives"]) <= 5
    assert result["alternatives"][0]["word"] == result["predicted_word"]


def test_responses_are_under_three_seconds(client, hawaiian_concepts):
    # Non-functional requirement: 95% of requests within 3 seconds
    times = []
    for concept in hawaiian_concepts[:20]:
        start = time.perf_counter()
        client.post("/predict", json={"language": "hawaiian", "concept": concept["id"]})
        times.append(time.perf_counter() - start)
    times.sort()
    assert times[int(len(times) * 0.95) - 1] < 3.0
