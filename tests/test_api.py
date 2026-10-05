from __future__ import annotations

from fastapi.testclient import TestClient

from api.index import app


client = TestClient(app)

REVIEWS = [
    "The food was flavorful and very well prepared.",
    "The biryani was delicious and portions were generous.",
    "Service was polite, attentive and reasonably quick.",
    "The ambience was elegant and the seating was comfortable.",
    "Prices are reasonable for the quality and portion size.",
    "The restaurant was clean and the washroom was hygienic.",
    "Overall it was a pleasant dining experience.",
    "The food quality was consistent across the dishes we ordered.",
    "The decor and atmosphere were welcoming.",
    "Good value for money for a family dinner.",
]


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "GastroEval API"}


def test_api_root() -> None:
    response = client.get("/api/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_deployment_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "GastroEval" in response.text
    assert 'id="analysis-form"' in response.text


def test_analyze_valid_request() -> None:
    response = client.post(
        "/api/analyze",
        json={"restaurant_name": "GastroEval Demo", "address": "Hyderabad", "reviews": REVIEWS},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(
        [
            "restaurant_name",
            "address",
            "gastroeval_score",
            "recommendation",
            "evidence_quality",
            "aspect_scores",
            "aspect_evidence_counts",
            "aspect_coverage",
            "strengths",
            "weaknesses",
            "limited_evidence",
            "representative_positive_evidence",
            "representative_negative_evidence",
            "metadata",
        ]
    ).issubset(body)
    assert body["restaurant_name"] == "GastroEval Demo"
    assert 0 <= body["gastroeval_score"] <= 100


def test_analyze_rejects_fewer_than_ten_reviews() -> None:
    response = client.post("/api/analyze", json={"restaurant_name": "Demo", "reviews": REVIEWS[:9]})
    assert response.status_code == 422
    assert "10" in response.json()["error"]


def test_analyze_rejects_empty_review() -> None:
    response = client.post(
        "/api/analyze",
        json={"restaurant_name": "Demo", "reviews": REVIEWS[:9] + ["  "]},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["error"].lower()


def test_analyze_rejects_malformed_request() -> None:
    response = client.post("/api/analyze", json={"reviews": REVIEWS})
    assert response.status_code == 422
    assert "restaurant_name" in response.json()["error"]
