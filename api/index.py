from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Vercel imports this module with the repository root as the function's parent,
# but local runners may start from another working directory. Derive all paths
# from this file instead of relying on cwd or a Windows-specific path.
API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gastroeval_engine import ASPECTS, analyze_reviews, load_inference_artifacts  # noqa: E402


LOGGER = logging.getLogger("gastroeval.api")
ARTIFACT_DIR = PROJECT_DIR / "gastroeval_artifacts"
REQUIRED_ARTIFACTS = (
    "tfidf_vectorizer.joblib",
    "sentiment_model.joblib",
    "aspect_weights.joblib",
    "aspect_inference_config.joblib",
    "scoring_config.joblib",
)


def _cors_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(
    title="GastroEval API",
    description="Explainable restaurant review evaluation powered by GastroEval.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    restaurant_name: str = Field(min_length=1)
    address: str = ""
    reviews: list[str] = Field(min_length=10)

    @field_validator("reviews")
    @classmethod
    def validate_reviews(cls, reviews: list[str]) -> list[str]:
        if not reviews:
            raise ValueError("At least 10 reviews are required.")
        if any(not review.strip() for review in reviews):
            raise ValueError("Reviews cannot be empty.")
        return reviews


class EvidenceItem(BaseModel):
    aspect: str
    sentiment: str
    evaluation_score: float | None = None
    confidence: float | None = None
    context: str


class AnalyzeResponse(BaseModel):
    restaurant_name: str
    address: str
    gastroeval_score: float
    recommendation: str
    evidence_quality: str
    aspect_scores: dict[str, float | None]
    aspect_evidence_counts: dict[str, int]
    aspect_coverage: float
    strengths: list[str]
    weaknesses: list[str]
    limited_evidence: list[str]
    representative_positive_evidence: list[EvidenceItem]
    representative_negative_evidence: list[EvidenceItem]
    metadata: dict[str, Any]


def _missing_artifacts() -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).is_file()]


def _json_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _evidence_items(evidence: pd.DataFrame, sentiment: str) -> list[EvidenceItem]:
    if evidence.empty:
        return []
    rows = evidence[evidence["Aspect_Sentiment"].astype(str) == sentiment].head(5)
    items: list[EvidenceItem] = []
    for _, row in rows.iterrows():
        items.append(
            EvidenceItem(
                aspect=str(row.get("Aspect", "")),
                sentiment=str(row.get("Aspect_Sentiment", sentiment)),
                evaluation_score=_json_value(row.get("Evaluation_Score")),
                confidence=_json_value(row.get("Sentiment_Confidence")),
                context=str(row.get("Aspect_Context", "")),
            )
        )
    return items


def _build_response(result: dict[str, Any], payload: AnalyzeRequest) -> AnalyzeResponse:
    summary = result["aspect_summary"]
    scores: dict[str, float | None] = {}
    evidence_counts: dict[str, int] = {}
    strengths: list[str] = []
    weaknesses: list[str] = []
    limited: list[str] = []

    for aspect in ASPECTS:
        rows = summary[summary["Aspect"].astype(str) == aspect]
        if rows.empty:
            scores[aspect] = None
            evidence_counts[aspect] = 0
            limited.append(aspect)
            continue
        row = rows.iloc[0]
        score = _json_value(row.get("Aspect_Score"))
        count = int(row.get("Evidence_Count", 0) or 0)
        scores[aspect] = score
        evidence_counts[aspect] = count
        if count == 0 or score is None:
            limited.append(aspect)
        elif score >= 70:
            strengths.append(aspect)
        elif score < 50:
            weaknesses.append(aspect)

    evidence = result["evidence"]
    return AnalyzeResponse(
        restaurant_name=payload.restaurant_name,
        address=payload.address,
        gastroeval_score=float(result["restaurant_score"]),
        recommendation=str(result["recommendation"]),
        evidence_quality=str(result["evidence_quality"]),
        aspect_scores=scores,
        aspect_evidence_counts=evidence_counts,
        aspect_coverage=float(result["coverage_pct"]),
        strengths=strengths,
        weaknesses=weaknesses,
        limited_evidence=limited,
        representative_positive_evidence=_evidence_items(evidence, "Positive"),
        representative_negative_evidence=_evidence_items(evidence, "Negative"),
        metadata={
            "review_count": len(payload.reviews),
            "evidence_count": int(result["evidence_count"]),
            "supported_aspect_count": len(ASPECTS) - len(limited),
            "aspects": ASPECTS,
            "inference_engine": "TF-IDF word n-grams + Logistic Regression + aspect evidence rules",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    messages = []
    for error in errors:
        location = error.get("loc", ())
        field = str(location[-1]) if location else "request"
        messages.append(f"{field}: {error.get('msg', 'Invalid request.')}")
    details = [
        {"loc": list(error.get("loc", ())), "msg": str(error.get("msg", "Invalid request.")), "type": error.get("type", "value_error")}
        for error in errors
    ]
    return JSONResponse(status_code=422, content={"error": "; ".join(messages), "details": details})


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled GastroEval API error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "The analysis could not be completed."})


@app.get("/api/")
async def api_root() -> dict[str, str]:
    return {"status": "ok", "service": "GastroEval API"}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "GastroEval API"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    missing = _missing_artifacts()
    if missing:
        LOGGER.error("Missing GastroEval artifacts: %s", ", ".join(missing))
        return JSONResponse(
            status_code=503,
            content={"error": "Required model artifacts are unavailable.", "missing_artifacts": missing},
        )  # type: ignore[return-value]

    try:
        # Warm/catch artifact loading separately so deployment errors identify
        # the exact stage without changing the engine's scoring behavior.
        load_inference_artifacts(str(ARTIFACT_DIR))
        result = analyze_reviews(
            payload.reviews,
            restaurant=payload.restaurant_name,
            artifact_dir=str(ARTIFACT_DIR),
        )
        return _build_response(result, payload)
    except FileNotFoundError as exc:
        LOGGER.exception("GastroEval artifact loading failed: %s", exc)
        return JSONResponse(status_code=503, content={"error": "Required model artifacts are unavailable."})  # type: ignore[return-value]
    except Exception as exc:
        LOGGER.exception("GastroEval inference failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Restaurant analysis failed."})  # type: ignore[return-value]
