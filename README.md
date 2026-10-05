GastroEval

Explainable ML-Based Restaurant Review Evaluation

GastroEval is a machine-learning system that converts customer-written restaurant reviews into an interpretable restaurant profile. Instead of requiring users to manually provide ratings or aspect labels, the application analyzes review text, detects important gastronomic dimensions, aggregates review-level evidence, and produces a restaurant-level score with traceable evidence.

The project is implemented as a Python + Streamlit application backed by a trained TF-IDF + Logistic Regression sentiment pipeline and an aspect-aware evidence/scoring engine.

Project Overview

A user provides:

Restaurant name

Restaurant address/context

At least 10 customer reviews, one review per line

GastroEval then:

Cleans and prepares the review text.

Predicts review sentiment using the trained ML model.

Detects gastronomic aspects mentioned in the reviews.

Builds aspect-level evidence from review contexts.

Aggregates unique review-aspect evidence units.

Calculates aspect scores on a 0-100 scale.

Combines supported aspects into an overall GastroEval score.

Shows representative evidence and evidence quality.

Provides downloadable CSV, HTML, and PDF reports when available.

The address is treated as contextual metadata and is not used as a model feature.

Six Evaluation Dimensions

GastroEval evaluates restaurants across six dimensions:

Dimension

Purpose

Food Quality

Quality, taste, preparation, freshness and food-related experience

Service

Staff behavior, responsiveness, waiting time and service quality

Ambience

Environment, seating, atmosphere, music, decor and comfort

Value for Money

Pricing, portions, affordability and perceived value

Hygiene

Cleanliness and hygiene-related evidence present in reviews

Overall Dining Experience

General customer experience and overall satisfaction

Missing aspects are treated as insufficient evidence rather than automatically receiving a zero score.

ML Pipeline

Sentiment Classification

The current trained sentiment pipeline uses:

TF-IDF word n-grams

Word n-gram range: 1-5

Logistic Regression classifier

Balanced class weighting

Class-probability-based evaluation scores

Phrase/rule enhancement for stronger domain-specific sentiment expressions

Small supervised TATVA augmentation set used for sentiment training

The prediction output is used as one source of evidence rather than being treated as the only signal.

Aspect Evidence Extraction

The evidence engine combines:

Restaurant-domain aspect triggers

Phrase matching

Clause-aware context extraction

Negation handling

Contrast handling

Sentiment probabilities

Rule-based phrase evidence

Multiple contexts from the same review and aspect are collapsed into a unique review-aspect evidence unit before restaurant-level aggregation. This prevents a single review from being over-counted simply because several matching phrases appear inside it.

Restaurant-Level Scoring

The default rubric weights are:

Dimension

Weight

Food Quality

30%

Service

20%

Ambience

15%

Value for Money

15%

Hygiene

10%

Overall Dining Experience

10%

Only supported dimensions contribute to the normalized final score. Missing dimensions are reported as insufficient evidence.

Model Evaluation

The restaurant-held-out evaluation protocol is designed to reduce restaurant-level leakage between training and testing.

Sentiment model

Metric

Validation

Held-out Test

Accuracy

83.96%

84.31%

Balanced Accuracy

77.62%

76.53%

Macro F1

75.45%

75.48%

Held-out restaurant score agreement

The end-to-end restaurant-level inference showed:

Pearson correlation: 0.818

Spearman correlation: 0.748

These figures describe agreement with the retained evaluation reference and should be interpreted as experimental evaluation metrics, not as a guarantee that every future restaurant will receive an equally accurate score.

Application Experience

The Streamlit application follows a minimal editorial interface rather than a dense dashboard.

Analyze

Users enter the restaurant information and paste reviews into a single review workspace.

Results

The results view contains:

Overall GastroEval score

Recommendation category

Evidence coverage and quality

Interactive vertical aspect graph for the six dimensions

Hover animation for the six aspect bars

Representative customer evidence

Downloadable result formats

Methodology

The methodology view explains the evaluation dimensions, model workflow and evidence aggregation approach.

Vercel Deployment
=================

GastroEval now has two intentionally separate entry points:

- Streamlit (`app.py`) remains the current standalone UI for local use.
- FastAPI (`api/index.py`) is the serverless backend for Vercel and reuses the existing `gastroeval_engine.py` inference path and trained artifacts.

The API does not retrain the model or change the GastroEval scoring methodology.

Architecture
------------

    Frontend or API client
            |
            v
    Vercel -> api/index.py (FastAPI)
            |
            v
    gastroeval_engine.py + gastroeval_artifacts/

Required production dependencies are listed in `requirements.txt`. Streamlit-only and report-generation packages remain in `requirements_streamlit.txt` and are not required by the API.

Local backend setup
-------------------

From the project directory:

    python -m venv .venv
    .venv\\Scripts\\activate       # Windows
    source .venv/bin/activate       # macOS/Linux
    pip install -r requirements.txt
    pip install pytest httpx

Run FastAPI locally:

    uvicorn api.index:app --reload

The API is available at `http://127.0.0.1:8000`.

Health checks
-------------

    curl http://127.0.0.1:8000/api/health
    curl http://127.0.0.1:8000/api/

Analyze request
---------------

    curl -X POST http://127.0.0.1:8000/api/analyze ^
      -H "Content-Type: application/json" ^
      -d "{\"restaurant_name\":\"GastroEval Demo\",\"address\":\"Hyderabad\",\"reviews\":[\"Excellent food and friendly service.\",\"The ambience was elegant.\",\"Good value for money.\",\"The restaurant was clean.\",\"The staff were attentive.\",\"The meal was flavorful.\",\"Portions were generous.\",\"The dining experience was pleasant.\",\"Service was quick.\",\"I would recommend this restaurant.\"]}"

The response includes the existing GastroEval score and recommendation, aspect scores and evidence counts, coverage, evidence-quality label, strengths, weaknesses, limited-evidence aspects, representative positive/negative evidence, and analysis metadata.

Environment variables
---------------------

`FRONTEND_ORIGIN` controls CORS. For local frontend development use:

    FRONTEND_ORIGIN=http://localhost:3000

For production, set it in Vercel to the deployed frontend origin. Multiple origins may be supplied as a comma-separated list.

Vercel deployment
-----------------

1. Push the repository to GitHub with `api/index.py`, `requirements.txt`, `vercel.json`, and `gastroeval_artifacts/` included.
2. In Vercel, choose **New Project** and import the repository.
3. Leave the framework preset as **Other** (or let Vercel detect the Python function).
4. Leave the root directory as the repository root.
5. Do not add a build command or a custom Node server.
6. Add `FRONTEND_ORIGIN` under Project Settings → Environment Variables.
7. Deploy and verify `/api/health`, then test `/api/analyze` with at least 10 non-empty reviews.

The `vercel.json` rewrite routes `/api/*` requests to the FastAPI function. The project must retain all five required joblib artifacts in `gastroeval_artifacts/`.

Testing
-------

Run the API tests with:

    pytest -q tests/test_api.py

The tests cover health endpoints, a valid model-backed analysis, validation failures, malformed requests, and response structure. They do not retrain the model.

Streamlit remains available locally with:

    streamlit run app.py

Deployment audit
----------------

The repository contains a large source dataset and the training notebook `GastroEval_Model.ipynb`. They are not needed by the production inference path and should be excluded from the Vercel deployment if they are not required by the Git history or another workflow. Do not remove them automatically. The trained artifacts are required and must remain available.

Interactive Aspect Graph

The results graph uses an interactive horizontal bar representation.

For each aspect, hovering over the bar reveals:

Aspect name

Score / 100

Evidence-unit count

The graph uses a fixed 0-100 scale so scores remain visually comparable across restaurants.

A replay/animation interaction is included to make the transition into the evaluated profile easier to read while keeping the graph lightweight.

Technology Stack

Python

Streamlit — application UI

Pandas — data handling

NumPy — numerical operations

scikit-learn — TF-IDF and Logistic Regression

SciPy — statistical utilities

Joblib — model/artifact serialization

Plotly — interactive result visualization

ReportLab — PDF report generation where enabled

Installation

1. Clone the repository

git clone https://github.com/MittaSaiVishruth/Restaurant-MLmodel.git
cd Restaurant-MLmodel

2. Create a virtual environment

py -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements_streamlit.txt

4. Run GastroEval

python -m streamlit run app.py

The application will open in the Streamlit browser interface.

Requirements

The application requires:

Python environment capable of running the listed dependencies

Streamlit

scikit-learn

pandas

numpy

scipy

joblib

plotly

reportlab (for PDF generation functionality)

The trained files inside gastroeval_artifacts/ must be available to the application.

Input Format

Enter one customer review per line.

Example:

The food was excellent and the portions were generous.
The service was slow but the staff were polite.
The ambience was peaceful and comfortable.
Prices were reasonable for the quantity served.
The dining area was clean and well maintained.
The biryani was flavorful and freshly prepared.
We had to wait a little longer than expected.
The seating area was comfortable.
Good value for money overall.
I would visit again.

Ratings are not required from the user.

Evidence and Explainability

GastroEval is designed so that an overall number is not presented in isolation.

The results are backed by evidence units that preserve information such as:

Source review

Review sentence/context

Detected aspect

Trigger phrase

ML sentiment

Model confidence

Evaluation score

Evidence decision source

This allows a reviewer or evaluator to trace how the restaurant profile was formed from the submitted reviews.

Data Quality Handling

The application can surface useful review-quality information such as:

Duplicate reviews

Very short/generic reviews

Sparse aspect evidence

Insufficient evidence for individual dimensions

Short reviews are not automatically discarded because some short reviews still contain useful sentiment or aspect information.

Reproducibility and Academic Use

The project is intended to support a structured restaurant-review research workflow rather than act as an authoritative real-world rating authority.

Important limitations include:

User-supplied reviews determine the evidence available to the system.

Sparse evidence can produce less reliable aspect scores.

Sentiment and aspect detection can make mistakes, especially on ambiguous or domain-specific language.

The restaurant-level score is a model-derived research output and should not be interpreted as a verified external restaurant rating.

Held-out evaluation reduces leakage risk, but model performance can vary on unseen restaurant domains and writing styles.
