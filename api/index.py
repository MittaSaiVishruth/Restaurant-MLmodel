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
from fastapi.responses import HTMLResponse, JSONResponse
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


WEB_APP = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#10120f" />
  <meta name="description" content="Turn restaurant reviews into an explainable, evidence-backed profile with GastroEval." />
  <title>GastroEval — Restaurant intelligence</title>
  <style>
    :root{color-scheme:dark;--bg:#10120f;--surface:#181b17;--surface2:#20241f;--text:#f4f3ed;--muted:#a7aa9f;--subtle:#777d71;--border:#30362e;--accent:#ed7b32;--accent2:#ffad69;--green:#8fc19a;--red:#e28e78;}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}button,input,textarea{font:inherit}button{cursor:pointer}.wrap{width:min(1080px,calc(100% - 44px));margin:auto}.topbar{height:76px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:11px}.mark{height:30px;width:30px;display:grid;place-items:center;border-radius:8px;background:var(--accent);color:#17140f;font-size:13px;font-weight:850}.brand-name{font-size:14px;font-weight:760;letter-spacing:-.02em}.brand-sub{margin-top:2px;color:var(--muted);font-size:10px}.status{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:11px}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px #8fc19a66}.hero{padding:58px 0 40px;max-width:720px}.eyebrow{color:var(--accent2);font-size:10px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}.hero h1{margin:13px 0 12px;font-size:clamp(36px,6vw,61px);line-height:1.02;letter-spacing:-.065em;font-weight:760}.hero p{max-width:600px;margin:0;color:var(--muted);font-size:15px;line-height:1.7}.layout{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(280px,.72fr);gap:26px;align-items:start}.panel{border:1px solid var(--border);background:var(--surface);padding:23px}.panel-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:18px}.panel-title{font-size:14px;font-weight:740}.panel-note{color:var(--subtle);font-size:10px}.field{margin:0 0 15px}.field label{display:block;margin:0 0 7px;color:#d9dbd3;font-size:11px;font-weight:650}.field input,.field textarea{display:block;width:100%;border:1px solid #363d34;border-radius:7px;background:#11140f;color:var(--text);outline:none;padding:11px 12px;font-size:12px;transition:border-color .18s,box-shadow .18s}.field input{height:42px}.field textarea{height:250px;resize:vertical;line-height:1.55}.field input:focus,.field textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px #ed7b321a}.field input::placeholder,.field textarea::placeholder{color:#70776b}.hint-row{display:flex;justify-content:space-between;gap:12px;margin-top:-7px;margin-bottom:13px;color:var(--subtle);font-size:10px}.hint-row strong{color:var(--muted);font-weight:650}.form-actions{display:flex;gap:9px;align-items:center}.primary,.secondary{min-height:42px;border-radius:7px;padding:0 15px;font-size:11px;font-weight:740;transition:background .18s,transform .18s,border-color .18s}.primary{border:1px solid var(--accent);background:var(--accent);color:#1b1712}.primary:hover{background:var(--accent2);border-color:var(--accent2);transform:translateY(-1px)}.primary:disabled{opacity:.58;cursor:wait;transform:none}.secondary{border:1px solid var(--border);background:transparent;color:var(--muted)}.secondary:hover{border-color:#5a6254;color:var(--text)}.intro{padding:5px 0}.intro h2{margin:0 0 9px;font-size:13px;font-weight:750}.intro>p{margin:0 0 20px;color:var(--muted);font-size:11px;line-height:1.7}.step{display:grid;grid-template-columns:28px 1fr;gap:11px;padding:13px 0;border-top:1px solid var(--border)}.step-no{color:var(--accent2);font:700 10px ui-monospace,monospace}.step strong{display:block;margin-bottom:4px;font-size:11px}.step span{color:var(--muted);font-size:10px;line-height:1.55}.trust{margin-top:18px;padding:13px;border-left:2px solid var(--accent);background:#1c1c17;color:var(--muted);font-size:10px;line-height:1.6}.error{display:none;margin-top:13px;padding:11px 12px;border:1px solid #753f32;background:#241815;color:#ffc4b0;font-size:11px;line-height:1.5}.loading{display:none;align-items:center;gap:10px;margin-top:14px;color:var(--muted);font-size:11px}.spinner{width:16px;height:16px;border:2px solid #393d33;border-top-color:var(--accent);border-radius:50%;animation:spin .75s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}#results{display:none;margin-top:48px;scroll-margin-top:24px}.result-heading{display:flex;justify-content:space-between;align-items:end;gap:18px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:20px 0;margin:13px 0 26px}.result-heading h2{margin:0 0 5px;font-size:24px;letter-spacing:-.04em}.result-address{color:var(--muted);font-size:11px}.scoreline{display:flex;align-items:baseline;gap:7px}.score{color:var(--text);font-size:47px;font-weight:780;line-height:1;letter-spacing:-.065em}.scoreunit{color:var(--muted);font-size:11px}.recommendation{padding:7px 10px;border:1px solid #714629;color:var(--accent2);font-size:10px;font-weight:750}.metrics{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:12px;color:var(--muted);font-size:10px}.metrics b{color:var(--text)}.section-title{margin:0 0 6px;font-size:13px;font-weight:750}.section-copy{margin:0 0 15px;color:var(--muted);font-size:10px;line-height:1.5}.chart{border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:16px 0;margin-bottom:28px}.aspect{display:grid;grid-template-columns:185px 1fr 44px;gap:12px;align-items:center;padding:8px 0}.aspect-label{color:#e6e7e0;font-size:10px}.bar-bg{height:7px;border-radius:5px;background:#282d26;overflow:hidden}.bar-fill{height:100%;border-radius:5px;background:var(--accent);transform-origin:left;animation:grow .75s cubic-bezier(.2,.8,.2,1) both;transition:filter .18s,transform .18s}.bar-bg:hover .bar-fill{filter:brightness(1.25);transform:scaleY(1.5)}@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}.aspect-score{text-align:right;color:var(--muted);font:11px ui-monospace,monospace}.aspect-score.na{font:10px system-ui,sans-serif;color:var(--subtle)}.result-columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.evidence-list{display:grid;gap:9px}.evidence-card{padding:12px;border:1px solid var(--border);background:var(--surface)}.evidence-top{display:flex;justify-content:space-between;gap:10px;margin-bottom:6px}.evidence-aspect{font-size:10px;font-weight:750}.sentiment{font-size:9px;font-weight:750}.sentiment.positive{color:var(--green)}.sentiment.negative{color:var(--red)}.evidence-text{color:#d4d7ce;font-size:10px;line-height:1.6}.empty-evidence{padding:13px;border:1px solid var(--border);color:var(--subtle);font-size:10px}.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:22px}.summary-card{padding:12px;border:1px solid var(--border);background:var(--surface)}.summary-card h3{margin:0 0 8px;color:var(--muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase}.summary-card p{margin:0;color:var(--text);font-size:10px;line-height:1.7}.footer{margin:54px 0 30px;padding-top:16px;border-top:1px solid var(--border);display:flex;justify-content:space-between;gap:12px;color:var(--subtle);font-size:9px}.footer a{color:var(--muted);text-decoration:none}.footer a:hover{color:var(--accent2)}
    @media(max-width:760px){.wrap{width:min(100% - 28px,1080px)}.topbar{height:64px}.hero{padding:42px 0 28px}.layout{grid-template-columns:1fr}.panel{padding:17px}.intro{display:none}.field textarea{height:225px}.result-heading{align-items:flex-start;flex-direction:column}.result-columns{grid-template-columns:1fr}.summary-grid{grid-template-columns:1fr}.aspect{grid-template-columns:125px 1fr 38px;gap:8px}.footer{flex-direction:column}.score{font-size:42px}}
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
  </style>
</head>
<body>
  <header class="topbar wrap"><div class="brand"><div class="mark">G</div><div><div class="brand-name">GastroEval</div><div class="brand-sub">Restaurant intelligence</div></div></div><div class="status"><span class="dot"></span>Review analysis</div></header>
  <main class="wrap">
    <section class="hero"><div class="eyebrow">Evidence led restaurant evaluation</div><h1>Reviews, made<br/>meaningful.</h1><p>Understand what diners say about the food, service, atmosphere and more. GastroEval turns customer reviews into a clear profile, with the evidence behind every score.</p></section>
    <section class="layout" aria-label="Restaurant review analysis">
      <form class="panel" id="analysis-form"><div class="panel-head"><div class="panel-title">Start an evaluation</div><div class="panel-note">At least 10 reviews</div></div>
        <div class="field"><label for="restaurant">Restaurant name</label><input id="restaurant" name="restaurant_name" required maxlength="120" placeholder="e.g. Beyond Flavours" /></div>
        <div class="field"><label for="address">Location <span style="color:var(--subtle);font-weight:450">· optional</span></label><input id="address" name="address" maxlength="240" placeholder="e.g. Hyderabad, Telangana" /></div>
        <div class="field"><label for="reviews">Customer reviews</label><textarea id="reviews" name="reviews" required placeholder="Paste reviews here, one review per line…"></textarea></div>
        <div class="hint-row"><span>One review per line</span><strong id="review-count">0 reviews</strong></div>
        <div class="form-actions"><button class="primary" id="analyze-button" type="submit">Analyze reviews <span aria-hidden="true">→</span></button><button class="secondary" type="button" id="example-button">Load example</button></div>
        <div class="error" id="error" role="alert"></div><div class="loading" id="loading" role="status"><span class="spinner"></span>Reading review evidence and evaluating six dimensions…</div>
      </form>
      <aside class="intro"><h2>From reviews to a reasoned view.</h2><p>GastroEval looks for review evidence across six parts of the dining experience. Supported dimensions contribute to the score; missing evidence is shown clearly.</p>
        <div class="step"><div class="step-no">01</div><div><strong>Share the reviews</strong><span>Paste 10 or more customer reviews, with one review on each line.</span></div></div>
        <div class="step"><div class="step-no">02</div><div><strong>See the dimensions</strong><span>Explore food quality, service, ambience, value, hygiene and overall experience.</span></div></div>
        <div class="step"><div class="step-no">03</div><div><strong>Check the evidence</strong><span>Review representative positive and negative excerpts behind the profile.</span></div></div>
        <div class="trust">Scores are based on the review evidence provided. They are an analytical guide, not a definitive restaurant grade.</div>
      </aside>
    </section>
    <section id="results" aria-live="polite"></section>
    <footer class="footer"><span>GastroEval · Explainable restaurant review analytics</span><a href="/api/health" target="_blank" rel="noreferrer">API status</a></footer>
  </main>
  <script>
    const exampleReviews=["The food was flavorful and very well prepared.","The biryani was delicious and portions were generous.","Service was polite, attentive and reasonably quick.","The ambience was elegant and the seating was comfortable.","Prices are reasonable for the quality and portion size.","The restaurant was clean and the washroom was hygienic.","Overall it was a pleasant dining experience and I would recommend it.","Food was excellent although the service was slightly slow.","The decor and atmosphere were welcoming.","The food quality was consistent across the dishes we ordered.","The waiting time was longer than expected.","Good value for money for a family dinner."];
    const form=document.getElementById('analysis-form'),reviewInput=document.getElementById('reviews'),countLabel=document.getElementById('review-count'),loading=document.getElementById('loading'),errorBox=document.getElementById('error'),submitButton=document.getElementById('analyze-button'),results=document.getElementById('results');
    const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    function getReviews(){return reviewInput.value.split(/\r?\n/).map(line=>line.trim()).filter(Boolean)}
    function refreshCount(){const n=getReviews().length;countLabel.textContent=`${n} ${n===1?'review':'reviews'}`;countLabel.style.color=n>=10?'var(--green)':'var(--muted)'}
    reviewInput.addEventListener('input',refreshCount);
    document.getElementById('example-button').addEventListener('click',()=>{reviewInput.value=exampleReviews.join('\n');refreshCount();reviewInput.focus()});
    function evidenceCards(items,type){if(!items?.length)return '<div class="empty-evidence">No representative '+type.toLowerCase()+' evidence was found.</div>';return '<div class="evidence-list">'+items.map(item=>`<article class="evidence-card"><div class="evidence-top"><span class="evidence-aspect">${escapeHtml(item.aspect)}</span><span class="sentiment ${type.toLowerCase()}">${type}</span></div><div class="evidence-text">“${escapeHtml(item.context)}”</div></article>`).join('')+'</div>'}
    function renderResults(data){const aspectEntries=Object.entries(data.aspect_scores||{});const bars=aspectEntries.map(([name,value],i)=>{const score=Number(value);const count=Number(data.aspect_evidence_counts?.[name]||0);const present=value!==null&&Number.isFinite(score);return `<div class="aspect"><div class="aspect-label">${escapeHtml(name)}</div><div class="bar-bg" title="${present?score.toFixed(1)+' out of 100 · '+count+' evidence units':'Insufficient evidence'}">${present?`<div class="bar-fill" style="width:${Math.max(0,Math.min(100,score))}%;animation-delay:${i*70}ms"></div>`:''}</div><div class="aspect-score ${present?'':'na'}">${present?score.toFixed(0):'No data'}</div></div>`}).join('');const summary=(title,values,empty)=>`<article class="summary-card"><h3>${title}</h3><p>${values?.length?values.map(escapeHtml).join('<br>'):empty}</p></article>`;results.innerHTML=`<div class="eyebrow">Evaluation result</div><div class="result-heading"><div><h2>${escapeHtml(data.restaurant_name)}</h2><div class="result-address">${escapeHtml(data.address||'Restaurant review analysis')}</div><div class="metrics"><span><b>${Number(data.metadata?.review_count||0)}</b> reviews</span><span><b>${Number(data.metadata?.evidence_count||0)}</b> evidence units</span><span><b>${Number(data.aspect_coverage||0).toFixed(0)}%</b> aspect coverage</span><span>${escapeHtml(data.evidence_quality)}</span></div></div><div><div class="scoreline"><div class="score">${Number(data.gastroeval_score).toFixed(1)}</div><div class="scoreunit">/ 100</div></div><div class="recommendation">${escapeHtml(data.recommendation)}</div></div></div><h3 class="section-title">Six dining dimensions</h3><p class="section-copy">Hover a bar for its score and evidence count. Missing evidence is not treated as a negative score.</p><div class="chart">${bars}</div><div class="summary-grid">${summary('Strengths',data.strengths,'No clear strengths detected')}${summary('Areas to consider',data.weaknesses,'No notable low scoring areas')}${summary('Limited evidence',data.limited_evidence,'All dimensions have evidence')}</div><div class="result-columns"><section><h3 class="section-title">Positive evidence</h3><p class="section-copy">Representative review excerpts supporting the profile.</p>${evidenceCards(data.representative_positive_evidence,'Positive')}</section><section><h3 class="section-title">Critical evidence</h3><p class="section-copy">Representative review excerpts that may point to issues.</p>${evidenceCards(data.representative_negative_evidence,'Negative')}</section></div>`;results.style.display='block';results.scrollIntoView({behavior:'smooth',block:'start'});}
    form.addEventListener('submit',async event=>{event.preventDefault();errorBox.style.display='none';results.style.display='none';const reviews=getReviews();if(reviews.length<10){errorBox.textContent=`Please add at least 10 non-empty reviews. You have ${reviews.length}.`;errorBox.style.display='block';return}loading.style.display='flex';submitButton.disabled=true;try{const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({restaurant_name:document.getElementById('restaurant').value.trim(),address:document.getElementById('address').value.trim(),reviews})});const data=await response.json().catch(()=>({error:'The server returned an unreadable response.'}));if(!response.ok)throw new Error(data.error||'Analysis failed. Please try again.');renderResults(data)}catch(err){errorBox.textContent=err.message||'Could not connect to GastroEval. Check your connection and try again.';errorBox.style.display='block'}finally{loading.style.display='none';submitButton.disabled=false}});
  </script>
</body>
</html>'''


@app.get("/api/")
async def api_root() -> dict[str, str]:
    return {"status": "ok", "service": "GastroEval API"}


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(WEB_APP)


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
