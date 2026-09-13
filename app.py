import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="GastroEval",
    page_icon="🍽️",
    layout="wide",
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "gastroeval_artifacts"

MODEL_FILE = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
SENTIMENT_FILE = ARTIFACT_DIR / "sentiment_model.joblib"
WEIGHTS_FILE = ARTIFACT_DIR / "aspect_weights.joblib"
SMOOTHING_FILE = ARTIFACT_DIR / "aspect_smoothing.joblib"

ASPECTS = [
    "Food Quality",
    "Service",
    "Ambience",
    "Value for Money",
    "Hygiene",
    "Overall Dining Experience",
]

# ---------------------------------------------------------
# Canonical aspect rules from the validated notebook
# ---------------------------------------------------------

ASPECT_TRIGGERS = {
    "Food Quality": [
        "food", "taste", "tasty", "delicious", "flavour", "flavor",
        "fresh", "stale", "food quality", "cooking", "cooked",
        "undercooked", "overcooked", "spicy", "salty", "oily", "bland",
        "dish", "dishes", "starter", "starters", "main course", "dessert",
        "desserts", "biryani", "pasta", "pizza",
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "server", "hospitality",
        "rude staff", "polite staff", "friendly staff", "courteous staff",
        "slow service", "quick service", "prompt service", "poor service",
        "bad service", "waiting", "waited", "manager", "bill",
    ],
    "Ambience": [
        "ambience", "ambiance", "atmosphere", "decor", "decoration",
        "interior", "lighting", "music", "seating", "chairs", "tables",
        "crowded", "quiet", "spacious", "cramped", "environment",
    ],
    "Value for Money": [
        "price", "pricing", "expensive", "overpriced", "affordable",
        "reasonable price", "reasonable pricing", "cheap", "value for money",
        "cost", "quantity", "portion", "worth the price", "price paid",
        "worth the money", "cost effective", "less quantity",
    ],
    "Hygiene": [
        "hygiene", "hygienic", "cleanliness", "dirty", "hair in",
        "hair strand", "insect", "cockroach", "fly in", "contaminated",
        "unclean", "poor hygiene", "unhygienic", "washroom", "toilet",
        "sanitary", "foul smell", "bad smell",
    ],
    "Overall Dining Experience": [
        "overall experience", "dining experience", "meal experience",
        "pleasant experience", "disappointing experience", "terrible experience",
        "wonderful experience", "awesome experience", "would recommend",
        "strongly recommend", "highly recommend", "must try", "must visit",
        "visit again", "go again", "not recommended",
    ],
}

ALL_EVALUATION_WORDS = [
    "good", "great", "excellent", "amazing", "awesome", "delicious", "tasty",
    "wonderful", "perfect", "fantastic", "best", "nice", "pleasant",
    "friendly", "courteous", "prompt", "quick", "reasonable", "affordable",
    "clean", "hygienic", "worth", "bad", "worst", "poor", "terrible",
    "horrible", "pathetic", "awful", "stale", "bland", "oily", "salty",
    "cold", "slow", "rude", "dirty", "expensive", "overpriced", "unhygienic",
    "disappointing", "disappointed", "not good", "not worth", "less quantity",
]

STRONG_NEGATIVE_PHRASES = [
    "not good", "not worth", "not worth the", "won't visit again",
    "wont visit again", "will not visit again", "never visit again",
    "worst food", "worst service", "very bad", "very poor", "very slow",
    "not satisfied", "didn't like", "didnt like", "do not recommend",
    "don't recommend", "dont recommend", "avoid this place", "waste of money",
    "not hygienic", "very unhygienic", "dirty washroom", "dirty toilet",
    "stale food", "food was stale",
]

STRONG_POSITIVE_PHRASES = [
    "very good", "very tasty", "very delicious", "very helpful", "very polite",
    "very courteous", "very prompt", "excellent food", "excellent service",
    "great food", "great service", "highly recommend", "must try", "must visit",
    "worth visiting", "worth the price", "good value", "very hygienic",
    "clean and hygienic", "will visit again", "would visit again",
]


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

@st.cache_resource

def load_artifacts():
    required = [MODEL_FILE, SENTIMENT_FILE, WEIGHTS_FILE, SMOOTHING_FILE]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
        )

    vectorizer = joblib.load(MODEL_FILE)
    sentiment_model = joblib.load(SENTIMENT_FILE)
    aspect_weights = joblib.load(WEIGHTS_FILE)
    smoothing_k = joblib.load(SMOOTHING_FILE)

    return vectorizer, sentiment_model, aspect_weights, smoothing_k


# ---------------------------------------------------------
# Canonical NLP/scoring functions
# ---------------------------------------------------------

def contains_pattern(text, pattern):
    text = str(text).lower()
    if " " in pattern:
        return pattern in text
    return re.search(rf"\b{re.escape(pattern)}\b", text) is not None


def split_into_sentences(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


def has_evaluation_context(sentence):
    text = sentence.lower()
    return any(
        contains_pattern(text, word)
        for word in ALL_EVALUATION_WORDS
    )


def detect_strong_aspects(sentence):
    text = str(sentence).strip()
    if not text:
        return []

    detected = []

    for aspect, triggers in ASPECT_TRIGGERS.items():
        aspect_found = any(
            contains_pattern(text, trigger)
            for trigger in triggers
        )

        evaluation_found = has_evaluation_context(text)

        if aspect == "Overall Dining Experience":
            if aspect_found:
                detected.append(aspect)
        elif aspect_found and evaluation_found:
            detected.append(aspect)

    return detected


def get_aspect_context(sentence, aspect, window=8):
    words = sentence.split()
    aspect_terms = aspect.lower().split()
    lower_words = [
        word.lower().strip(".,!?;:()[]{}")
        for word in words
    ]

    for i in range(len(lower_words) - len(aspect_terms) + 1):
        if lower_words[i:i + len(aspect_terms)] == aspect_terms:
            start = max(0, i - window)
            end = min(len(words), i + len(aspect_terms) + window)
            return " ".join(words[start:end])

    return sentence


def adjusted_sentiment(context, ml_prediction):
    text = str(context).lower()

    for phrase in STRONG_NEGATIVE_PHRASES:
        if phrase in text:
            return "Negative"

    for phrase in STRONG_POSITIVE_PHRASES:
        if phrase in text:
            return "Positive"

    return ml_prediction


def recommendation_label(score):
    if score >= 85:
        return "Highly Recommended"
    if score >= 75:
        return "Recommended"
    if score >= 65:
        return "Generally Recommended"
    if score >= 50:
        return "Consider with Caution"
    return "Not Recommended"


def evaluate_restaurant_reviews(
    restaurant_name,
    address,
    reviews,
    vectorizer,
    sentiment_model,
    aspect_weights,
    smoothing_k,
):
    if not isinstance(reviews, (list, tuple, pd.Series)):
        raise TypeError("reviews must be a list, tuple, or pandas Series.")

    cleaned_reviews = []
    for review in reviews:
        if pd.isna(review):
            continue
        review = str(review).strip()
        if review:
            cleaned_reviews.append(review)

    if len(cleaned_reviews) < 10:
        raise ValueError("Please provide at least 10 non-empty reviews.")

    evidence_records = []

    for review in cleaned_reviews:
        for sentence in split_into_sentences(review):
            detected_aspects = detect_strong_aspects(sentence)
            for aspect in detected_aspects:
                context = get_aspect_context(sentence, aspect)
                evidence_records.append({
                    "Restaurant": restaurant_name,
                    "Address": address,
                    "Review": review,
                    "Sentence": sentence,
                    "Aspect": aspect,
                    "Aspect_Context": context,
                })

    if not evidence_records:
        return {
            "Restaurant": restaurant_name,
            "Address": address,
            "Total_Reviews": len(cleaned_reviews),
            "Total_Aspect_Evidence": 0,
            "GastroEval_Score": 50.0,
            "Recommendation": "Insufficient Evidence",
            "Aspect_Coverage_Pct": 0.0,
            "Aspect_Scores": {a: 50.0 for a in ASPECTS},
            "Evidence_Counts": {a: 0 for a in ASPECTS},
            "Evidence_Confidence": {a: 0.0 for a in ASPECTS},
            "Evidence_Detail": pd.DataFrame(),
            "Aspect_Summary": pd.DataFrame(),
        }

    evidence_df = pd.DataFrame(evidence_records)
    tfidf = vectorizer.transform(evidence_df["Aspect_Context"])
    evidence_df["ML_Sentiment"] = sentiment_model.predict(tfidf)
    evidence_df["Aspect_Sentiment"] = evidence_df.apply(
        lambda row: adjusted_sentiment(
            row["Aspect_Context"], row["ML_Sentiment"]
        ),
        axis=1,
    )

    summary = (
        evidence_df
        .groupby("Aspect")
        .agg(
            Evidence_Count=("Aspect_Sentiment", "size"),
            Negative=("Aspect_Sentiment", lambda x: (x == "Negative").sum()),
            Neutral=("Aspect_Sentiment", lambda x: (x == "Neutral").sum()),
            Positive=("Aspect_Sentiment", lambda x: (x == "Positive").sum()),
        )
        .reindex(aspect_weights.keys())
        .fillna(0)
        .reset_index()
    )

    denominator = summary["Evidence_Count"].replace(0, np.nan)
    summary["Positive_Rate"] = summary["Positive"] / denominator
    summary["Neutral_Rate"] = summary["Neutral"] / denominator
    summary["Negative_Rate"] = summary["Negative"] / denominator

    summary["Aspect_Score"] = (
        summary["Positive_Rate"].fillna(0)
        + 0.5 * summary["Neutral_Rate"].fillna(0)
    ) * 100

    summary["Evidence_Confidence"] = np.where(
        summary["Evidence_Count"] > 0,
        summary["Evidence_Count"]
        / (summary["Evidence_Count"] + smoothing_k),
        0,
    )

    summary["Adjusted_Aspect_Score"] = np.where(
        summary["Evidence_Count"] > 0,
        (
            summary["Aspect_Score"] * summary["Evidence_Confidence"]
            + 50 * (1 - summary["Evidence_Confidence"])
        ),
        50,
    )

    summary["Base_Weight"] = summary["Aspect"].map(aspect_weights)
    summary["Effective_Weight"] = (
        summary["Base_Weight"] * summary["Evidence_Confidence"]
    )

    total_effective_weight = summary["Effective_Weight"].sum()
    if total_effective_weight == 0:
        final_score = 50.0
        summary["Normalized_Weight"] = 0.0
        summary["Weighted_Contribution"] = 0.0
    else:
        summary["Normalized_Weight"] = (
            summary["Effective_Weight"] / total_effective_weight
        )
        summary["Weighted_Contribution"] = (
            summary["Adjusted_Aspect_Score"]
            * summary["Normalized_Weight"]
        )
        final_score = summary["Weighted_Contribution"].sum()

    aspects_with_evidence = (summary["Evidence_Count"] > 0).sum()
    coverage = aspects_with_evidence / len(aspect_weights) * 100

    aspect_scores = {
        row["Aspect"]: round(float(row["Adjusted_Aspect_Score"]), 2)
        for _, row in summary.iterrows()
    }
    evidence_counts = {
        row["Aspect"]: int(row["Evidence_Count"])
        for _, row in summary.iterrows()
    }
    evidence_confidence = {
        row["Aspect"]: round(float(row["Evidence_Confidence"] * 100), 2)
        for _, row in summary.iterrows()
    }

    return {
        "Restaurant": restaurant_name,
        "Address": address,
        "Total_Reviews": len(cleaned_reviews),
        "Total_Aspect_Evidence": len(evidence_df),
        "GastroEval_Score": round(float(final_score), 2),
        "Recommendation": recommendation_label(final_score),
        "Aspect_Coverage_Pct": round(float(coverage), 2),
        "Aspect_Scores": aspect_scores,
        "Evidence_Counts": evidence_counts,
        "Evidence_Confidence": evidence_confidence,
        "Evidence_Detail": evidence_df,
        "Aspect_Summary": summary,
    }


def interpret_aspect(score, evidence_count, min_evidence=5):
    if evidence_count < min_evidence:
        return "Insufficient Evidence"
    if score >= 70:
        return "Strength"
    if score < 60:
        return "Needs Attention"
    return "Acceptable"


def build_aspect_table(result, min_evidence=5):
    rows = []
    for aspect in ASPECTS:
        score = result["Aspect_Scores"][aspect]
        evidence = result["Evidence_Counts"][aspect]
        confidence = result["Evidence_Confidence"][aspect]
        status = interpret_aspect(score, evidence, min_evidence)
        rows.append({
            "Aspect": aspect,
            "Score": score,
            "Evidence": evidence,
            "Confidence": confidence,
            "Status": status,
        })
    return pd.DataFrame(rows)


def representative_evidence(result, max_per_sentiment=2):
    evidence_df = result["Evidence_Detail"]
    output = {}

    if evidence_df.empty:
        return {a: {"Positive": [], "Negative": []} for a in ASPECTS}

    for aspect in ASPECTS:
        rows = evidence_df[evidence_df["Aspect"] == aspect]
        output[aspect] = {
            "Positive": rows.loc[
                rows["Aspect_Sentiment"] == "Positive", "Aspect_Context"
            ].drop_duplicates().head(max_per_sentiment).tolist(),
            "Negative": rows.loc[
                rows["Aspect_Sentiment"] == "Negative", "Aspect_Context"
            ].drop_duplicates().head(max_per_sentiment).tolist(),
        }
    return output


def build_explanation(result, min_evidence=5):
    aspect_table = build_aspect_table(result, min_evidence)
    strengths = aspect_table[aspect_table["Status"] == "Strength"].sort_values(
        "Score", ascending=False
    )
    needs = aspect_table[aspect_table["Status"] == "Needs Attention"].sort_values(
        "Score", ascending=True
    )
    limited = aspect_table[aspect_table["Status"] == "Insufficient Evidence"]["Aspect"].tolist()

    return {
        "strengths": strengths.to_dict("records"),
        "needs": needs.to_dict("records"),
        "limited": limited,
        "aspect_table": aspect_table,
        "evidence": representative_evidence(result),
    }


# ---------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------

def build_pdf_report(result, explanation):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"GastroEval Report - {result['Restaurant']}",
        author="GastroEval",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], alignment=TA_CENTER,
        spaceAfter=8, fontSize=20,
    )
    h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], spaceBefore=10,
        spaceAfter=5, fontSize=13,
    )
    body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontSize=9,
        leading=12, spaceAfter=4,
    )
    small = ParagraphStyle(
        "SmallCustom", parent=body, fontSize=8, leading=10,
    )

    story = []
    story.append(Paragraph("GastroEval Restaurant Evaluation Report", title_style))
    story.append(Paragraph(f"<b>Restaurant:</b> {html.escape(result['Restaurant'])}", body))
    story.append(Paragraph(f"<b>Address:</b> {html.escape(result['Address'])}", body))
    story.append(Paragraph(
        f"<b>Reviews analysed:</b> {result['Total_Reviews']} &nbsp;&nbsp; "
        f"<b>Aspect evidence:</b> {result['Total_Aspect_Evidence']}",
        body,
    ))
    story.append(Spacer(1, 4))

    summary_data = [
        ["GastroEval Score", f"{result['GastroEval_Score']:.2f} / 100"],
        ["Recommendation", result["Recommendation"]],
        ["Aspect Coverage", f"{result['Aspect_Coverage_Pct']:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[48 * mm, 120 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Aspect Evaluation", h2))
    aspect_rows = [["Aspect", "Score", "Evidence", "Confidence", "Status"]]
    for row in explanation["aspect_table"].to_dict("records"):
        aspect_rows.append([
            row["Aspect"],
            f"{row['Score']:.2f}",
            str(row["Evidence"]),
            f"{row['Confidence']:.2f}%",
            row["Status"],
        ])

    aspect_table = Table(
        aspect_rows,
        colWidths=[55 * mm, 24 * mm, 24 * mm, 28 * mm, 38 * mm],
        repeatRows=1,
    )
    aspect_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(aspect_table)

    story.append(Paragraph("Strengths", h2))
    if explanation["strengths"]:
        for row in explanation["strengths"]:
            story.append(Paragraph(
                f"<b>{html.escape(row['Aspect'])}</b>: {row['Score']:.2f}/100 "
                f"({row['Evidence']} evidence records)", body
            ))
    else:
        story.append(Paragraph("No aspect met the strength threshold.", body))

    story.append(Paragraph("Needs Attention", h2))
    if explanation["needs"]:
        for row in explanation["needs"]:
            story.append(Paragraph(
                f"<b>{html.escape(row['Aspect'])}</b>: {row['Score']:.2f}/100 "
                f"({row['Evidence']} evidence records)", body
            ))
    else:
        story.append(Paragraph("No sufficiently supported aspect was classified as needs attention.", body))

    story.append(Paragraph("Insufficient Evidence", h2))
    if explanation["limited"]:
        story.append(Paragraph(
            html.escape(", ".join(explanation["limited"])), body
        ))
    else:
        story.append(Paragraph("None.", body))

    story.append(PageBreak())
    story.append(Paragraph("Representative Review Evidence", h2))

    for aspect in ASPECTS:
        evidence = explanation["evidence"][aspect]
        story.append(Paragraph(f"<b>{html.escape(aspect)}</b>", body))
        for sentiment_label in ["Positive", "Negative"]:
            snippets = evidence[sentiment_label]
            if snippets:
                story.append(Paragraph(
                    f"<b>{sentiment_label} evidence:</b>", small
                ))
                for snippet in snippets:
                    safe = html.escape(str(snippet))
                    story.append(Paragraph(f"- {safe}", small))
            
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Method note: GastroEval is an evidence-aware analytical score derived from "
        "aspect-level sentiment. Insufficient evidence is not treated as poor performance.",
        small,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_html_report(result, explanation):
    def esc(value):
        return html.escape(str(value))

    rows = "".join(
        f"<tr><td>{esc(row['Aspect'])}</td><td>{row['Score']:.2f}</td>"
        f"<td>{row['Evidence']}</td><td>{row['Confidence']:.2f}%</td>"
        f"<td>{esc(row['Status'])}</td></tr>"
        for row in explanation["aspect_table"].to_dict("records")
    )

    strengths = "".join(
        f"<li><b>{esc(x['Aspect'])}</b>: {x['Score']:.2f}/100 "
        f"({x['Evidence']} evidence records)</li>"
        for x in explanation["strengths"]
    ) or "<li>None</li>"

    needs = "".join(
        f"<li><b>{esc(x['Aspect'])}</b>: {x['Score']:.2f}/100 "
        f"({x['Evidence']} evidence records)</li>"
        for x in explanation["needs"]
    ) or "<li>None</li>"

    limited = ", ".join(map(esc, explanation["limited"])) or "None"

    evidence_sections = []
    for aspect in ASPECTS:
        evidence = explanation["evidence"][aspect]
        blocks = [f"<h4>{esc(aspect)}</h4>"]
        for label in ["Positive", "Negative"]:
            if evidence[label]:
                blocks.append(f"<b>{label} evidence</b><ul>")
                blocks.extend(f"<li>{esc(x)}</li>" for x in evidence[label])
                blocks.append("</ul>")
        if not evidence["Positive"] and not evidence["Negative"]:
            blocks.append("<p>No representative evidence available.</p>")
        evidence_sections.append("".join(blocks))

    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>GastroEval - {esc(result['Restaurant'])}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 36px; color: #172033; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; }}
.meta {{ color: #536174; margin-bottom: 18px; }}
.hero {{ display: flex; gap: 18px; margin: 20px 0; }}
.card {{ border: 1px solid #d9dee7; border-radius: 10px; padding: 16px; min-width: 180px; }}
.score {{ font-size: 28px; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #d9dee7; padding: 8px; text-align: left; }}
th {{ background: #eef2f7; }}
.note {{ background: #f7f9fb; padding: 12px; border-left: 4px solid #94a3b8; }}
</style>
</head>
<body>
<h1>GastroEval Restaurant Evaluation Report</h1>
<div class='meta'><b>Restaurant:</b> {esc(result['Restaurant'])}<br>
<b>Address:</b> {esc(result['Address'])}<br>
<b>Reviews analysed:</b> {result['Total_Reviews']} &nbsp; | &nbsp;
<b>Aspect evidence:</b> {result['Total_Aspect_Evidence']}</div>

<div class='hero'>
<div class='card'><div>GastroEval Score</div><div class='score'>{result['GastroEval_Score']:.2f}/100</div></div>
<div class='card'><div>Recommendation</div><div class='score'>{esc(result['Recommendation'])}</div></div>
<div class='card'><div>Aspect Coverage</div><div class='score'>{result['Aspect_Coverage_Pct']:.2f}%</div></div>
</div>

<h2>Aspect Evaluation</h2>
<table><thead><tr><th>Aspect</th><th>Score</th><th>Evidence</th><th>Confidence</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>Strengths</h2><ul>{strengths}</ul>
<h2>Needs Attention</h2><ul>{needs}</ul>
<h2>Insufficient Evidence</h2><p>{limited}</p>

<h2>Representative Review Evidence</h2>
{''.join(evidence_sections)}

<p class='note'><b>Method note:</b> GastroEval is an evidence-aware analytical score derived from aspect-level sentiment. Insufficient evidence is not treated as poor performance.</p>
</body></html>"""


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("🍽️ GastroEval")
st.caption(
    "Explainable ML-based restaurant evaluation from restaurant name, address, and review evidence."
)

with st.expander("How GastroEval works", expanded=False):
    st.write(
        "Enter the restaurant name and address, then provide at least 10 reviews. "
        "The trained NLP pipeline extracts evidence for six gastronomic aspects, "
        "predicts aspect-level sentiment, adjusts for explicit sentiment phrases, "
        "accounts for evidence volume, and produces an evidence-aware 0-100 score."
    )

try:
    vectorizer, sentiment_model, aspect_weights, smoothing_k = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Place the four model files in a folder named 'gastroeval_artifacts' next to app.py."
    )
    st.stop()

with st.form("restaurant_form"):
    restaurant_name = st.text_input(
        "Restaurant name",
        placeholder="e.g. Beyond Flavours",
    )
    address = st.text_input(
        "Restaurant address",
        placeholder="e.g. Gachibowli, Hyderabad",
    )
    reviews_text = st.text_area(
        "Restaurant reviews (minimum 10; one review per line)",
        height=320,
        placeholder=(
            "The food was excellent and the service was prompt.\n"
            "Ambience was beautiful and comfortable.\n"
            "The price was reasonable for the quantity.\n"
            "..."
        ),
    )

    submit = st.form_submit_button(
        "Generate GastroEval Report",
        type="primary",
        use_container_width=True,
    )

if submit:
    if not restaurant_name.strip():
        st.warning("Please enter the restaurant name.")
        st.stop()
    if not address.strip():
        st.warning("Please enter the restaurant address.")
        st.stop()

    # One review per non-empty line.
    reviews = [line.strip() for line in reviews_text.splitlines() if line.strip()]

    if len(reviews) < 10:
        st.warning(
            f"Please provide at least 10 non-empty reviews. You provided {len(reviews)}."
        )
        st.stop()

    with st.spinner("Analysing reviews and preparing the report..."):
        result = evaluate_restaurant_reviews(
            restaurant_name=restaurant_name.strip(),
            address=address.strip(),
            reviews=reviews,
            vectorizer=vectorizer,
            sentiment_model=sentiment_model,
            aspect_weights=aspect_weights,
            smoothing_k=smoothing_k,
        )

        explanation = build_explanation(result)
        pdf_bytes = build_pdf_report(result, explanation)
        html_report = build_html_report(result, explanation)

    st.session_state["latest_result"] = result
    st.session_state["latest_explanation"] = explanation
    st.session_state["latest_pdf"] = pdf_bytes
    st.session_state["latest_html"] = html_report


# ---------------------------------------------------------
# Report display
# ---------------------------------------------------------

if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    explanation = st.session_state["latest_explanation"]

    st.divider()
    st.header("GastroEval Report")

    c1, c2, c3 = st.columns(3)
    c1.metric("GastroEval Score", f"{result['GastroEval_Score']:.2f} / 100")
    c2.metric("Recommendation", result["Recommendation"])
    c3.metric("Aspect Coverage", f"{result['Aspect_Coverage_Pct']:.2f}%")

    st.caption(
        f"{result['Total_Reviews']} reviews analysed | "
        f"{result['Total_Aspect_Evidence']} aspect evidence records"
    )

    st.subheader("Aspect Evaluation")
    aspect_table = explanation["aspect_table"].copy()
    st.dataframe(
        aspect_table,
        use_container_width=True,
        hide_index=True,
    )

    chart_df = aspect_table.set_index("Aspect")[["Score"]]
    st.bar_chart(chart_df)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Strengths")
        if explanation["strengths"]:
            for item in explanation["strengths"]:
                st.success(
                    f"{item['Aspect']}: {item['Score']:.2f}/100 "
                    f"({item['Evidence']} evidence records)"
                )
        else:
            st.info("No sufficiently supported strength identified.")

    with col_b:
        st.subheader("Needs Attention")
        if explanation["needs"]:
            for item in explanation["needs"]:
                st.warning(
                    f"{item['Aspect']}: {item['Score']:.2f}/100 "
                    f"({item['Evidence']} evidence records)"
                )
        else:
            st.info("No sufficiently supported aspect needs attention.")

    st.subheader("Insufficient Evidence")
    if explanation["limited"]:
        st.info(
            "No strong conclusion is made for: "
            + ", ".join(explanation["limited"])
        )
    else:
        st.success("All six aspects have at least the minimum evidence threshold.")

    st.subheader("Representative Review Evidence")

    for aspect in ASPECTS:
        evidence = explanation["evidence"][aspect]
        with st.expander(aspect):
            if evidence["Positive"]:
                st.markdown("**Positive evidence**")
                for snippet in evidence["Positive"]:
                    st.write(f"- {snippet}")
            if evidence["Negative"]:
                st.markdown("**Negative evidence**")
                for snippet in evidence["Negative"]:
                    st.write(f"- {snippet}")
            if not evidence["Positive"] and not evidence["Negative"]:
                st.caption("No representative evidence available.")

    st.subheader("Download Report")

    d1, d2 = st.columns(2)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", result["Restaurant"]).strip("_") or "restaurant"

    with d1:
        st.download_button(
            "Download PDF Report",
            data=st.session_state["latest_pdf"],
            file_name=f"GastroEval_{safe_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download HTML Report",
            data=st.session_state["latest_html"],
            file_name=f"GastroEval_{safe_name}.html",
            mime="text/html",
            use_container_width=True,
        )

    st.caption(
        "Method note: GastroEval is an evidence-aware analytical score. "
        "Insufficient evidence is not automatically treated as poor performance."
    )
