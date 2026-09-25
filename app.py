from __future__ import annotations

import html
import inspect
import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from gastroeval_engine import analyze_reviews


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
# Keep Streamlit's native shell quiet. The visual system below owns the UI.
st.set_page_config(
    page_title="GastroEval",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "gastroeval_artifacts"


# -----------------------------------------------------------------------------
# Design tokens
# -----------------------------------------------------------------------------
BG = "#F7F6F2"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F1F0EB"
TEXT = "#17202A"
MUTED = "#64748B"
SUBTLE = "#94A3B8"
BORDER = "#E5E7EB"
ACCENT = "#EA580C"
ACCENT_DARK = "#C2410C"
SUCCESS = "#355C4A"
DANGER = "#9A3412"

NAV_ITEMS = (
    ("analyze", "Analyze"),
    ("results", "Results"),
    ("methodology", "Methodology"),
)

REQUIRED_ARTIFACTS = (
    "tfidf_vectorizer.joblib",
    "sentiment_model.joblib",
    "aspect_weights.joblib",
    "aspect_inference_config.joblib",
    "scoring_config.joblib",
)

DEMO_REVIEWS = [
    "The food was flavorful and very well prepared.",
    "The biryani was delicious and portions were generous.",
    "Service was polite, attentive and reasonably quick.",
    "The ambience was elegant and the seating was comfortable.",
    "Prices are reasonable for the quality and portion size.",
    "The restaurant was clean and the washroom was hygienic.",
    "Overall it was a pleasant dining experience and I would recommend it.",
    "Food was excellent although the service was slightly slow.",
    "The decor and atmosphere were welcoming.",
    "The food quality was consistent across the dishes we ordered.",
    "The waiting time was longer than expected.",
    "Good value for money for a family dinner.",
]


# -----------------------------------------------------------------------------
# Minimal CSS
# -----------------------------------------------------------------------------
# IMPORTANT: this is a normal triple-quoted string, not an f-string. That avoids
# the CSS-brace/f-string failure that affected previous revisions.
CSS = f"""
<style>
:root {{
    color-scheme: light !important;
    --ge-bg: {BG};
    --ge-surface: {SURFACE};
    --ge-surface-alt: {SURFACE_ALT};
    --ge-text: {TEXT};
    --ge-muted: {MUTED};
    --ge-subtle: {SUBTLE};
    --ge-border: {BORDER};
    --ge-accent: {ACCENT};
    --ge-accent-dark: {ACCENT_DARK};
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{
    background: var(--ge-bg) !important;
    color: var(--ge-text) !important;
}}

body {{
    color-scheme: light !important;
}}

/* Native shell */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {{
    display: none !important;
}}

/* Main width and rhythm */
.block-container {{
    width: min(1060px, calc(100vw - 44px)) !important;
    max-width: 1060px !important;
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}

/* Prevent Streamlit's container layers from introducing dark/white panels. */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],
[data-testid="stForm"] {{
    color: var(--ge-text) !important;
}}

/* Generic text */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: var(--ge-text) !important;
}}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: var(--ge-muted) !important;
}}

/* Brand */
.ge-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.25rem;
}}

.ge-mark {{
    width: 27px;
    height: 27px;
    display: grid;
    place-items: center;
    border-radius: 7px;
    background: var(--ge-text);
    color: #fff !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: -.02em;
}}

.ge-brand-name {{
    color: var(--ge-text) !important;
    font-size: .94rem;
    line-height: 1.1;
    font-weight: 760;
    letter-spacing: -.02em;
}}

.ge-brand-sub {{
    margin-top: 2px;
    color: var(--ge-muted) !important;
    font-size: .72rem;
    line-height: 1.1;
}}

/* Minimal button navigation */
.ge-nav {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 22px;
    margin-bottom: 3rem;
    border-bottom: 1px solid var(--ge-border);
}}

.ge-nav-spacer {{
    height: 1px;
}}

[data-testid="stButton"] {{
    margin: 0 !important;
}}

[data-testid="stButton"] button {{
    min-height: 42px !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--ge-muted) !important;
    box-shadow: none !important;
    font-size: .79rem !important;
    font-weight: 650 !important;
    padding: 0 2px 11px !important;
    transition: color .16s ease, border-color .16s ease, background .16s ease !important;
}}

[data-testid="stButton"] button:hover {{
    color: var(--ge-text) !important;
    background: transparent !important;
}}

/* Navigation button keys */
.st-key-nav_analyze button,
.st-key-nav_results button,
.st-key-nav_methodology button {{
    width: 100% !important;
}}

.ge-nav-active button {{
    color: var(--ge-text) !important;
    border-bottom: 2px solid var(--ge-accent) !important;
}}

/* Primary CTA */
.st-key-submit_analysis button {{
    min-height: 44px !important;
    padding: 0 20px !important;
    border-radius: 8px !important;
    background: var(--ge-accent) !important;
    border: 1px solid var(--ge-accent) !important;
    color: #fff !important;
    font-size: .82rem !important;
    font-weight: 700 !important;
}}

.st-key-submit_analysis button:hover {{
    background: var(--ge-accent-dark) !important;
    border-color: var(--ge-accent-dark) !important;
}}

/* Secondary actions */
.st-key-load_demo button,
.st-key-clear_analysis button,
.st-key-edit_results button {{
    min-height: 40px !important;
    padding: 0 14px !important;
    border: 1px solid var(--ge-border) !important;
    border-radius: 8px !important;
    background: var(--ge-surface) !important;
    color: var(--ge-text) !important;
    font-size: .78rem !important;
}}

.st-key-load_demo button:hover,
.st-key-clear_analysis button:hover,
.st-key-edit_results button:hover {{
    border-color: #CBD5E1 !important;
    background: var(--ge-surface-alt) !important;
}}

/* Inputs */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stFileUploader"] label,
[data-testid="stSelectbox"] label {{
    color: var(--ge-text) !important;
    font-size: .76rem !important;
    font-weight: 650 !important;
}}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background: var(--ge-surface) !important;
    color: var(--ge-text) !important;
    border: 1px solid var(--ge-border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}

[data-testid="stTextInput"] input {{
    min-height: 44px !important;
}}

[data-testid="stTextArea"] textarea {{
    min-height: 250px !important;
    line-height: 1.55 !important;
}}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {{
    color: var(--ge-subtle) !important;
    opacity: 1 !important;
}}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: #CBD5E1 !important;
    box-shadow: 0 0 0 2px rgba(234, 88, 12, .08) !important;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--ge-surface) !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    color: var(--ge-muted) !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    background: var(--ge-surface) !important;
    color: var(--ge-text) !important;
    border: 1px solid var(--ge-border) !important;
    border-radius: 7px !important;
    min-height: 36px !important;
    box-shadow: none !important;
}}

[data-testid="stFileUploaderDropzone"] button:hover {{
    background: var(--ge-surface-alt) !important;
    border-color: #CBD5E1 !important;
}}

/* Selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background: var(--ge-surface) !important;
    color: var(--ge-text) !important;
    border-color: var(--ge-border) !important;
    border-radius: 8px !important;
}}

/* Form surface */
[data-testid="stForm"] {{
    border: 0 !important;
    background: transparent !important;
    padding: 0 !important;
}}

/* Hero */
.ge-eyebrow {{
    margin-bottom: .55rem;
    color: var(--ge-accent-dark) !important;
    font-size: .67rem;
    line-height: 1.2;
    font-weight: 760;
    letter-spacing: .12em;
    text-transform: uppercase;
}}

.ge-title {{
    max-width: 760px;
    margin-bottom: .65rem;
    color: var(--ge-text) !important;
    font-size: clamp(2.1rem, 4vw, 3.25rem);
    line-height: 1.03;
    font-weight: 760;
    letter-spacing: -.045em;
}}

.ge-lead {{
    max-width: 760px;
    margin-bottom: 2.6rem;
    color: var(--ge-muted) !important;
    font-size: .98rem;
    line-height: 1.6;
}}

.ge-section {{
    padding-top: 1.8rem;
    margin-top: 1.8rem;
    border-top: 1px solid var(--ge-border);
}}

.ge-section-title {{
    margin-bottom: .35rem;
    color: var(--ge-text) !important;
    font-size: .88rem;
    font-weight: 760;
    letter-spacing: -.01em;
}}

.ge-section-copy {{
    max-width: 760px;
    margin-bottom: 1rem;
    color: var(--ge-muted) !important;
    font-size: .76rem;
    line-height: 1.55;
}}

.ge-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    margin: .65rem 0 1rem;
    color: var(--ge-muted);
    font-size: .72rem;
}}

.ge-meta b {{
    color: var(--ge-text);
}}

.ge-note {{
    margin: 1rem 0;
    padding: 10px 12px;
    border-left: 2px solid #CBD5E1;
    background: rgba(255,255,255,.55);
    color: var(--ge-muted) !important;
    font-size: .75rem;
    line-height: 1.55;
}}

/* Result summary */
.ge-result-head {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    padding: 1.2rem 0 1.45rem;
    border-top: 1px solid var(--ge-border);
    border-bottom: 1px solid var(--ge-border);
    margin-top: 1.4rem;
    margin-bottom: 2rem;
}}

.ge-score-line {{
    display: flex;
    align-items: baseline;
    gap: 10px;
}}

.ge-score {{
    color: var(--ge-text) !important;
    font-size: clamp(3.2rem, 7vw, 5rem);
    line-height: .9;
    font-weight: 780;
    letter-spacing: -.065em;
}}

.ge-score-unit {{
    color: var(--ge-muted) !important;
    font-size: .8rem;
    font-weight: 650;
}}

.ge-recommendation {{
    padding-bottom: 6px;
    color: var(--ge-accent-dark) !important;
    font-size: .82rem;
    font-weight: 760;
    white-space: nowrap;
}}

/* Aspect list */
.ge-aspect-list {{
    border-top: 1px solid var(--ge-border);
}}

.ge-aspect-row {{
    padding: 14px 0 15px;
    border-bottom: 1px solid var(--ge-border);
}}

.ge-aspect-top {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 20px;
}}

.ge-aspect-name {{
    color: var(--ge-text) !important;
    font-size: .8rem;
    font-weight: 700;
}}

.ge-aspect-meta {{
    margin-top: 3px;
    color: var(--ge-subtle) !important;
    font-size: .68rem;
}}

.ge-aspect-score {{
    color: var(--ge-text) !important;
    font-size: .92rem;
    font-weight: 760;
}}

.ge-bar {{
    height: 3px;
    margin-top: 10px;
    overflow: hidden;
    border-radius: 3px;
    background: var(--ge-surface-alt);
}}

.ge-bar span {{
    display: block;
    height: 100%;
    border-radius: inherit;
    background: var(--ge-accent);
}}

/* Native aspect rows */
.ge-native-aspect-name {{
    margin-top: 2px;
    color: var(--ge-text) !important;
    font-size: .8rem;
    font-weight: 700;
}}

.ge-native-aspect-score {{
    padding-top: 4px;
    color: var(--ge-text) !important;
    font-size: .95rem;
    font-weight: 780;
    text-align: right;
}}

.ge-missing-list {{
    margin-top: 10px;
    padding: 10px 12px;
    border-left: 2px solid var(--ge-border);
    color: var(--ge-muted) !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .68rem;
    line-height: 1.7;
}}

div[data-testid="stProgress"] > div > div {{
    height: 3px !important;
    border-radius: 999px !important;
}}

div[data-testid="stProgress"] div[role="progressbar"] {{
    min-height: 3px !important;
    border-radius: 999px !important;
}}

/* Evidence */
.ge-evidence {{
    padding: 14px 0;
    border-bottom: 1px solid var(--ge-border);
}}

.ge-evidence-kicker {{
    margin-bottom: 5px;
    color: var(--ge-accent-dark) !important;
    font-size: .67rem;
    font-weight: 760;
}}

.ge-evidence-text {{
    color: var(--ge-text) !important;
    font-size: .81rem;
    line-height: 1.55;
}}

.ge-evidence-meta {{
    margin-top: 7px;
    color: var(--ge-subtle) !important;
    font-size: .67rem;
}}

/* Methodology */
.ge-method-row {{
    display: grid;
    grid-template-columns: 44px 190px minmax(0, 1fr);
    gap: 18px;
    padding: 15px 0;
    border-top: 1px solid var(--ge-border);
}}

.ge-method-row:last-child {{
    border-bottom: 1px solid var(--ge-border);
}}

.ge-method-no {{
    color: var(--ge-accent-dark) !important;
    font-size: .69rem;
    font-weight: 760;
}}

.ge-method-title {{
    color: var(--ge-text) !important;
    font-size: .79rem;
    font-weight: 730;
}}

.ge-method-body {{
    color: var(--ge-muted) !important;
    font-size: .75rem;
    line-height: 1.55;
}}

/* Dividers and dataframe */
hr {{
    border: 0 !important;
    border-top: 1px solid var(--ge-border) !important;
    margin: 2rem 0 !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--ge-border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}}

/* Spinner */
[data-testid="stSpinner"] * {{
    color: var(--ge-muted) !important;
}}

/* Error/info/success messages */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
}}

/* Tabs: keep the evidence workspace minimal and always readable. */
[data-baseweb="tab-list"] {{
    gap: 18px !important;
    border-bottom: 1px solid var(--ge-border) !important;
}}

[data-baseweb="tab"] {{
    color: var(--ge-muted) !important;
    font-size: .78rem !important;
    font-weight: 650 !important;
    padding: 10px 2px !important;
}}

[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--ge-text) !important;
}}

[data-baseweb="tab-highlight"] {{
    background: var(--ge-accent) !important;
    height: 2px !important;
}}

/* Download controls: readable before hover, not dark/inverted. */
[data-testid="stDownloadButton"] {{
    width: 100% !important;
    margin: 0 !important;
}}

[data-testid="stDownloadButton"] button {{
    width: 100% !important;
    min-height: 44px !important;
    border: 1px solid var(--ge-border) !important;
    border-radius: 8px !important;
    background: var(--ge-surface) !important;
    color: var(--ge-text) !important;
    box-shadow: none !important;
    font-size: .78rem !important;
    font-weight: 680 !important;
    transition: background .16s ease, border-color .16s ease, transform .16s ease !important;
}}

[data-testid="stDownloadButton"] button:hover {{
    background: var(--ge-surface-alt) !important;
    border-color: #CBD5E1 !important;
    color: var(--ge-text) !important;
}}

[data-testid="stDownloadButton"] button:active {{
    transform: translateY(1px);
}}

[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span {{
    color: var(--ge-text) !important;
}}

.ge-export-type {{
    color: var(--ge-accent-dark) !important;
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.ge-export-copy {{
    color: var(--ge-muted) !important;
    font-size: .69rem;
    line-height: 1.45;
    min-height: 2.2em;
    margin-bottom: 9px;
}}

/* Plotly aspect chart */
.ge-chart-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 18px;
    margin-top: 1rem;
}}

.ge-chart-head .ge-chart-note {{
    margin: 0;
}}

.ge-chart-shell {{
    margin: 1rem 0 1.35rem;
    padding: 0.35rem 0 0.15rem;
    border-top: 1px solid var(--ge-border);
    border-bottom: 1px solid var(--ge-border);
}}

.ge-chart-note {{
    margin-top: -0.3rem;
    margin-bottom: 0.75rem;
    color: var(--ge-subtle) !important;
    font-size: .67rem;
    line-height: 1.5;
}}

/* Interactive aspect bar chart */
.ge-bar-shell {{
    margin-top: .95rem;
}}

.ge-bar-shell .stPlotlyChart {{
    width: 100% !important;
}}

/* Mobile */
@media (max-width: 760px) {{
    .block-container {{
        width: min(100% - 28px, 1060px) !important;
        padding-top: 1.4rem !important;
    }}

    .ge-nav {{
        gap: 8px;
        margin-bottom: 2.2rem;
    }}

    .ge-title {{
        font-size: 2.2rem;
    }}

    .ge-result-head {{
        align-items: flex-start;
        flex-direction: column;
    }}

    .ge-export-copy {{
        min-height: auto;
    }}

    .ge-recommendation {{
        padding-bottom: 0;
    }}

    .ge-method-row {{
        grid-template-columns: 36px 1fr;
    }}

    .ge-method-body {{
        grid-column: 2;
    }}
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        scroll-behavior: auto !important;
        transition: none !important;
        animation: none !important;
    }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
# Widget values live in their own keys. Application state uses different keys.
# This separation is intentional: Streamlit forbids changing a widget key after
# that widget has already been instantiated in the same run.
def init_state() -> None:
    defaults = {
        "view": "analyze",
        "input_restaurant_name": "",
        "input_address": "",
        "input_review_text": "",
        "analysis_restaurant": "",
        "analysis_address": "",
        "analysis_reviews": [],
        "result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# One-time cleanup for sessions created by older GastroEval builds.
# st.file_uploader is a button-like widget whose key is read-only; an old
# session can still contain the legacy key from previous versions.
APP_STATE_VERSION = 10
if st.session_state.get("_ge_app_state_version") != APP_STATE_VERSION:
    st.session_state.pop("review_upload", None)
    st.session_state["_ge_app_state_version"] = APP_STATE_VERSION

init_state()


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def esc(value: Any) -> str:
    return html.escape(str(value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _supports_parameter(function: Any, parameter: str) -> bool:
    """Return whether the installed Streamlit API exposes a parameter."""
    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _width_kwargs(function: Any, stretch: bool) -> dict[str, Any]:
    """Use the current width API, with compatibility for older Streamlit builds."""
    if _supports_parameter(function, "width"):
        return {"width": "stretch" if stretch else "content"}
    return {"use_container_width": stretch}


def ge_button(label: str, **kwargs: Any) -> bool:
    """Version-safe Streamlit button wrapper."""
    kwargs.update(_width_kwargs(st.button, kwargs.pop("_stretch", False)))
    return st.button(label, **kwargs)


def ge_form_submit_button(label: str, **kwargs: Any) -> bool:
    """Version-safe form submit button wrapper."""
    kwargs.update(_width_kwargs(st.form_submit_button, kwargs.pop("_stretch", False)))
    return st.form_submit_button(label, **kwargs)


def ge_plotly_chart(figure: Any, **kwargs: Any) -> Any:
    """Version-safe interactive Plotly renderer."""
    kwargs.update(_width_kwargs(st.plotly_chart, kwargs.pop("_stretch", True)))
    return st.plotly_chart(figure, **kwargs)


def ge_dataframe(data: Any, **kwargs: Any) -> Any:
    """Version-safe dataframe renderer."""
    kwargs.update(_width_kwargs(st.dataframe, kwargs.pop("_stretch", True)))
    return st.dataframe(data, **kwargs)


def split_reviews(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def review_quality_stats(reviews: list[str]) -> dict[str, int]:
    normalized = [" ".join(str(r).split()).strip().lower() for r in reviews]
    normalized = [r for r in normalized if r]
    unique = len(set(normalized))
    return {
        "count": len(normalized),
        "unique": unique,
        "duplicates": len(normalized) - unique,
        "short": sum(len(r.split()) <= 3 for r in normalized),
    }


def artifact_status() -> tuple[bool, list[str]]:
    missing = [
        filename
        for filename in REQUIRED_ARTIFACTS
        if not (ARTIFACT_DIR / filename).exists()
    ]
    return not missing, missing


def goto_view(view: str) -> None:
    if view == "results" and st.session_state.result is None:
        st.session_state.view = "analyze"
    else:
        st.session_state.view = view


def clear_all() -> None:
    """Callback: safe place to reset widget state before the next rerun."""
    st.session_state.input_restaurant_name = ""
    st.session_state.input_address = ""
    st.session_state.input_review_text = ""
    st.session_state.analysis_restaurant = ""
    st.session_state.analysis_address = ""
    st.session_state.analysis_reviews = []
    st.session_state.result = None
    st.session_state.view = "analyze"


def load_demo() -> None:
    """Callback: runs before the form widgets are instantiated on the rerun."""
    st.session_state.input_restaurant_name = "GastroEval Demo Restaurant"
    st.session_state.input_address = "Hyderabad, Telangana"
    st.session_state.input_review_text = "\n".join(DEMO_REVIEWS)
    st.session_state.result = None
    st.session_state.view = "analyze"


def handle_upload() -> None:
    """Callback for TXT/CSV import. The textarea has not been instantiated yet."""
    uploaded_file = st.session_state.get("review_upload")
    if uploaded_file is None:
        return

    try:
        name = str(getattr(uploaded_file, "name", "")).lower()
        if name.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        else:
            frame = pd.read_csv(uploaded_file)
            if frame.empty or len(frame.columns) == 0:
                content = ""
            else:
                candidates = [
                    column
                    for column in frame.columns
                    if any(
                        token in str(column).lower()
                        for token in ("review", "text", "comment")
                    )
                ]
                column = candidates[0] if candidates else frame.columns[0]
                content = "\n".join(frame[column].dropna().astype(str).tolist())
        st.session_state.input_review_text = content
    except Exception as exc:
        st.session_state.upload_error = f"Could not import the file: {exc}"


def clear_upload_error() -> None:
    st.session_state.pop("upload_error", None)


def render_nav() -> None:
    """Render navigation with callback-only application state.

    No navigation widget is backed by a mutable session-state key. This keeps
    navigation safe across repeated Analyze -> Methodology -> Analyze -> Results
    transitions and avoids StreamlitWidgetAlreadyInstantiatedError.
    """
    view = str(st.session_state.get("view", "analyze"))
    has_result = st.session_state.get("result") is not None
    active_key = f"nav_{view}"

    st.markdown(
        f"""
        <style>
        .st-key-nav_analyze button,
        .st-key-nav_results button,
        .st-key-nav_methodology button {{
            color: {MUTED} !important;
        }}
        .st-key-{active_key} button {{
            color: {TEXT} !important;
            border-bottom: 2px solid {ACCENT} !important;
        }}
        .st-key-nav_results button[disabled] {{
            opacity: .42 !important;
            cursor: default !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        ge_button(
            "Analyze", key="nav_analyze", type="secondary",
            _stretch=True, on_click=goto_view, args=("analyze",)
        )
    with c2:
        ge_button(
            "Results", key="nav_results", type="secondary",
            _stretch=True, disabled=not has_result,
            on_click=goto_view, args=("results",)
        )
    with c3:
        ge_button(
            "Methodology", key="nav_methodology", type="secondary",
            _stretch=True, on_click=goto_view, args=("methodology",)
        )


def render_brand() -> None:
    st.markdown(
        """
        <div class="ge-brand">
            <div class="ge-mark">G</div>
            <div>
                <div class="ge-brand-name">GastroEval</div>
                <div class="ge-brand-sub">Restaurant intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_note(reviews: list[str]) -> None:
    stats = review_quality_stats(reviews)
    if stats["duplicates"] > 0:
        st.info(
            f"{stats['duplicates']} duplicate review"
            f"{'s' if stats['duplicates'] != 1 else ''} detected. "
            "Repeated text can increase the influence of the same opinion."
        )
    if stats["count"] and stats["short"] >= max(3, int(stats["count"] * 0.25)):
        st.info(
            f"{stats['short']} reviews are three words or fewer. Very short text can reduce aspect-level evidence coverage."
        )


def score_bar(score: Any) -> str:
    score = safe_float(score, float("nan"))
    if pd.isna(score):
        return "<div class='ge-bar'></div>"
    score = max(0.0, min(100.0, score))
    return f"<div class='ge-bar'><span style='width:{score:.1f}%;'></span></div>"


def render_aspect_chart(aspect_summary: pd.DataFrame) -> None:
    """Render the six aspect scores as an interactive, animated horizontal bar chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("The interactive aspect graph needs Plotly. Install the project requirements and restart Streamlit.")
        fallback = aspect_summary[["Aspect", "Aspect_Score", "Evidence_Count"]].copy()
        fallback.columns = ["Aspect", "Score", "Evidence units"]
        ge_dataframe(fallback, _stretch=True, hide_index=True)
        return

    frame = aspect_summary.copy()
    frame["Aspect"] = frame["Aspect"].astype(str)
    frame["Score"] = pd.to_numeric(frame["Aspect_Score"], errors="coerce")
    frame["Evidence"] = pd.to_numeric(frame["Evidence_Count"], errors="coerce").fillna(0).astype(int)

    canonical = [
        "Food Quality",
        "Service",
        "Ambience",
        "Value for Money",
        "Hygiene",
        "Overall Dining Experience",
    ]
    order_map = {name: i for i, name in enumerate(canonical)}
    frame["_order"] = frame["Aspect"].map(order_map).fillna(999)
    frame = frame.sort_values("_order")

    supported = frame[frame["Score"].notna()].copy()
    missing = frame.loc[frame["Score"].isna(), "Aspect"].tolist()

    if supported.empty:
        st.markdown(
            '<div class="ge-chart-shell"><div class="ge-chart-note">No supported aspects were detected from the submitted reviews.</div></div>',
            unsafe_allow_html=True,
        )
        return

    customdata = supported[["Evidence"]].to_numpy()
    scores = supported["Score"].clip(0, 100).astype(float).round(1).tolist()
    aspects = supported["Aspect"].tolist()

    # Frames provide a lightweight entrance animation and a visible Replay control.
    # The chart itself remains fully usable even when reduced motion is preferred.
    steps = 12
    frames = []
    for i in range(steps + 1):
        progress = i / steps
        frames.append(
            go.Frame(
                name=f"aspect_frame_{i}",
                data=[
                    go.Bar(
                        x=[round(value * progress, 1) for value in scores],
                        y=aspects,
                        customdata=customdata,
                        orientation="h",
                        marker=dict(
                            color=ACCENT,
                            line=dict(color=ACCENT_DARK, width=0.6),
                        ),
                        text=[f"{value:.0f}" for value in scores],
                        textposition="outside",
                        textfont=dict(color=TEXT, size=12),
                        cliponaxis=False,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Score: %{x:.1f}/100<br>"
                            "Evidence units: %{customdata[0]}<br>"
                            "<span style='color:#EA580C'>Hover to inspect this dimension</span>"
                            "<extra></extra>"
                        ),
                        name="Aspect score",
                    )
                ],
            )
        )

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=aspects,
                orientation="h",
                customdata=customdata,
                marker=dict(
                    color=ACCENT,
                    line=dict(color=ACCENT_DARK, width=0.6),
                ),
                text=[f"{v:.0f}" for v in scores],
                textposition="outside",
                textfont=dict(color=TEXT, size=12),
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Score: %{x:.1f}/100<br>"
                    "Evidence units: %{customdata[0]}<br>"
                    "<span style='color:#EA580C'>Hover to inspect this dimension</span>"
                    "<extra></extra>"
                ),
                name="Aspect score",
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        height=max(390, 70 * len(supported) + 90),
        margin=dict(l=8, r=58, t=28, b=44),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        showlegend=False,
        hovermode="closest",
        hoverdistance=20,
        bargap=0.32,
        transition=dict(duration=450, easing="cubic-in-out"),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
            color=TEXT,
            size=12,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0,
                xanchor="left",
                y=1.06,
                yanchor="bottom",
                pad=dict(r=4, t=2, b=2, l=0),
                buttons=[
                    dict(
                        label="Replay",
                        method="animate",
                        args=[
                            [f"aspect_frame_{i}" for i in range(steps + 1)],
                            {
                                "frame": {"duration": 55, "redraw": True},
                                "transition": {"duration": 35, "easing": "cubic-in-out"},
                                "fromcurrent": False,
                                "mode": "immediate",
                            },
                        ],
                    )
                ],
                font=dict(size=10, color=TEXT),
                bgcolor=SURFACE,
                bordercolor=BORDER,
                borderwidth=1,
            )
        ],
        xaxis=dict(
            range=[0, 105],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50", "75", "100"],
            title="Score",
            title_font=dict(color=MUTED, size=11),
            tickfont=dict(color=MUTED, size=10),
            gridcolor=BORDER,
            gridwidth=1,
            zeroline=False,
            linecolor=BORDER,
            fixedrange=True,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=aspects[::-1],
            tickfont=dict(color=TEXT, size=11),
            gridcolor="rgba(0,0,0,0)",
            fixedrange=True,
            automargin=True,
        ),
        hoverlabel=dict(
            bgcolor=TEXT,
            bordercolor=ACCENT,
            font=dict(color="#FFFFFF", size=12),
            align="left",
            namelength=-1,
        ),
    )

    st.markdown('<div class="ge-chart-shell ge-bar-shell">', unsafe_allow_html=True)
    ge_plotly_chart(
        fig,
        _stretch=True,
        theme=None,
        config={
            "displaylogo": False,
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
        },
        key="aspect_profile_bar",
    )

    note = "Hover a bar for its score and evidence count. Use Replay to animate the profile."
    if missing:
        note += " Insufficient evidence: " + ", ".join(missing) + "."
    st.markdown(f'<div class="ge-chart-note">{esc(note)}</div></div>', unsafe_allow_html=True)

def render_aspect_rows(aspect_summary: pd.DataFrame) -> None:
    """Accessible native fallback for environments without Plotly."""
    for _, row in aspect_summary.iterrows():
        aspect = str(row.get("Aspect", ""))
        score = safe_float(row.get("Aspect_Score"), float("nan"))
        count = int(row.get("Evidence_Count", 0))
        left, right = st.columns([8.7, 1.3], gap="small")
        with left:
            st.markdown(
                f"<div class='ge-native-aspect-name'>{esc(aspect)}</div>",
                unsafe_allow_html=True,
            )
            if pd.isna(score):
                st.caption("Insufficient evidence")
                progress_value = 0
            else:
                unit = "evidence unit" if count == 1 else "evidence units"
                st.caption(f"{count} {unit}")
                progress_value = int(round(max(0.0, min(100.0, score))))
            st.progress(progress_value, text=None)
        with right:
            display_score = "—" if pd.isna(score) else f"{score:.0f}"
            st.markdown(
                f"<div class='ge-native-aspect-score'>{display_score}</div>",
                unsafe_allow_html=True,
            )


def render_evidence(evidence: pd.DataFrame, limit: int = 8) -> None:
    if evidence.empty:
        st.markdown(
            "<div class='ge-note'>No aspect-level evidence was detected in the submitted reviews.</div>",
            unsafe_allow_html=True,
        )
        return

    view = evidence.sort_values("Evaluation_Score", ascending=False).head(limit)
    for _, row in view.iterrows():
        sentiment = str(row.get("Aspect_Sentiment", "Neutral"))
        label = {
            "Positive": "Positive evidence",
            "Negative": "Negative evidence",
            "Neutral": "Neutral evidence",
        }.get(sentiment, "Evidence")
        st.markdown(
            f"""
            <div class="ge-evidence">
                <div class="ge-evidence-kicker">{esc(row.get('Aspect', ''))} · {esc(label)}</div>
                <div class="ge-evidence-text">“{esc(row.get('Aspect_Context', row.get('Review', '')))}”</div>
                <div class="ge-evidence-meta">
                    Evaluation score {safe_float(row.get('Evaluation_Score')):.1f}
                    · confidence {safe_float(row.get('Sentiment_Confidence')):.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_html_report(result: dict, restaurant: str, address: str) -> bytes:
    aspect_rows: list[str] = []
    for _, row in result["aspect_summary"].iterrows():
        score = "N/A" if pd.isna(row["Aspect_Score"]) else f"{safe_float(row['Aspect_Score']):.1f}"
        aspect_rows.append(
            f"<tr><td>{esc(row['Aspect'])}</td><td>{score}</td><td>{int(row['Evidence_Count'])}</td></tr>"
        )

    evidence_rows: list[str] = []
    for _, row in result["evidence"].head(12).iterrows():
        evidence_rows.append(
            f"<div class='evidence'><strong>{esc(row['Aspect'])}</strong>"
            f"<span>{esc(row['Aspect_Sentiment'])} · {safe_float(row['Evaluation_Score']):.1f}</span>"
            f"<p>{esc(row['Aspect_Context'])}</p></div>"
        )

    markup = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GastroEval Report — {esc(restaurant)}</title>
<style>
body{{font-family:Arial,sans-serif;background:{BG};color:{TEXT};padding:40px;line-height:1.5}}
main{{max-width:900px;margin:0 auto}}
.kicker{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:{ACCENT_DARK};font-weight:700}}
h1{{font-size:38px;line-height:1.05;margin:8px 0 8px}}
.meta{{color:{MUTED};font-size:14px;margin-bottom:28px}}
.score{{font-size:64px;font-weight:800;letter-spacing:-.05em;margin:20px 0 0}}
.rec{{color:{ACCENT_DARK};font-weight:700;margin-bottom:28px}}
table{{width:100%;border-collapse:collapse;background:{SURFACE}}}
th,td{{padding:11px 12px;border-bottom:1px solid {BORDER};text-align:left;font-size:13px}}
th{{font-size:12px;color:{MUTED}}}
.evidence{{padding:14px 0;border-bottom:1px solid {BORDER}}}
.evidence span{{color:{MUTED};font-size:12px;margin-left:8px}}
.evidence p{{margin:5px 0 0;font-size:14px}}
</style>
</head>
<body><main>
<div class="kicker">GastroEval</div>
<h1>{esc(restaurant)}</h1>
<div class="meta">{esc(address)}</div>
<div class="score">{safe_float(result['restaurant_score']):.1f}<span style="font-size:20px;color:{MUTED}"> / 100</span></div>
<div class="rec">{esc(result['recommendation'])}</div>
<p class="meta">{len(result['evidence'])} evidence units · {safe_float(result['coverage_pct']):.0f}% aspect coverage · {esc(result['evidence_quality'])}</p>
<table><thead><tr><th>Aspect</th><th>Score</th><th>Evidence</th></tr></thead><tbody>{''.join(aspect_rows)}</tbody></table>
<h2>Representative evidence</h2>
{''.join(evidence_rows) if evidence_rows else '<p class="meta">No evidence available.</p>'}
</main></body></html>"""
    return markup.encode("utf-8")


def build_pdf_report(result: dict, restaurant: str, address: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GastroEval", styles["Title"]),
        Paragraph("Explainable restaurant evaluation", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(f"<b>Restaurant:</b> {esc(restaurant)}", styles["BodyText"]),
        Paragraph(f"<b>Address:</b> {esc(address)}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            f"<b>GastroEval score:</b> {safe_float(result['restaurant_score']):.2f}/100",
            styles["BodyText"],
        ),
        Paragraph(
            f"<b>Recommendation:</b> {esc(result['recommendation'])}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    data = [["Aspect", "Score", "Evidence"]]
    for _, row in result["aspect_summary"].iterrows():
        score = "N/A" if pd.isna(row["Aspect_Score"]) else f"{safe_float(row['Aspect_Score']):.1f}"
        data.append([str(row["Aspect"]), score, str(int(row["Evidence_Count"]))])

    table = Table(data, colWidths=[250, 80, 80])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEXT)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(BORDER)),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    story.extend([table, Spacer(1, 14), Paragraph("Representative evidence", styles["Heading2"])])
    for _, row in result["evidence"].head(8).iterrows():
        story.append(
            Paragraph(
                f"<b>{esc(row['Aspect'])}</b> — {esc(row['Aspect_Sentiment'])} — "
                f"{safe_float(row['Evaluation_Score']):.1f}<br/>"
                f"{esc(row['Aspect_Context'])}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


def run_analysis(name: str, address: str, reviews: list[str]) -> dict:
    return analyze_reviews(
        reviews,
        restaurant=name,
        artifact_dir=ARTIFACT_DIR,
    )


# -----------------------------------------------------------------------------
# Shell
# -----------------------------------------------------------------------------
render_brand()
render_nav()


# -----------------------------------------------------------------------------
# Analyze view
# -----------------------------------------------------------------------------
if st.session_state.view == "analyze":
    st.markdown('<div class="ge-eyebrow">Explainable restaurant intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="ge-title">Turn reviews into a clear restaurant profile.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ge-lead">Analyze customer review text across six gastronomic aspects and produce an evidence-backed restaurant score.</div>',
        unsafe_allow_html=True,
    )

    ready, missing = artifact_status()

    # Utility row is intentionally small and secondary.
    st.markdown('<div class="ge-section-title">Restaurant details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ge-section-copy">The name identifies the restaurant. The address is contextual metadata and is not used as a model feature.</div>',
        unsafe_allow_html=True,
    )

    utility_left, utility_mid, utility_right = st.columns([1.3, 1.0, 1.0], gap="small")
    with utility_left:
        st.file_uploader(
            "Import reviews",
            type=["txt", "csv"],
            key="review_upload",
            on_change=handle_upload,
            help="TXT: one review per line. CSV: a review/text/comment column is preferred.",
        )
    with utility_mid:
        st.write("")
        ge_button(
            "Load example",
            key="load_demo",
            _stretch=True,
            on_click=load_demo,
        )
    with utility_right:
        stats_before = review_quality_stats(split_reviews(st.session_state.input_review_text))
        st.markdown(
            f"<div class='ge-meta' style='justify-content:flex-end;padding-top:10px;'>"
            f"<span><b>{stats_before['count']}</b> reviews</span>"
            f"<span><b>{stats_before['unique']}</b> unique</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if "upload_error" in st.session_state:
        st.error(st.session_state.upload_error)
        clear_upload_error()

    # Form is deliberate: typing does not rerun the app, and all form inputs
    # are read together only when the user submits the analysis.
    with st.form("analysis_form", clear_on_submit=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.text_input(
                "Restaurant name",
                key="input_restaurant_name",
                placeholder="e.g. Beyond Flavours",
            )
        with right:
            st.text_input(
                "Address",
                key="input_address",
                placeholder="e.g. Hyderabad, Telangana",
            )

        st.markdown('<div style="height:7px"></div>', unsafe_allow_html=True)
        st.text_area(
            "Customer reviews",
            key="input_review_text",
            height=285,
            placeholder="Paste one review per line…",
            help="Minimum 10 reviews. Ratings are not required.",
        )

        st.markdown(
            '<div class="ge-section-copy" style="margin-top:7px;margin-bottom:0;">Minimum 10 reviews · one review per line · ratings are not required</div>',
            unsafe_allow_html=True,
        )

        submit = ge_form_submit_button(
            "Analyze restaurant",
            type="primary",
            _stretch=False,
            key="submit_analysis",
        )

    current_reviews = split_reviews(st.session_state.input_review_text)
    current_stats = review_quality_stats(current_reviews)
    if current_reviews:
        st.markdown(
            f"<div class='ge-meta'>"
            f"<span><b>{current_stats['count']}</b> reviews</span>"
            f"<span><b>{current_stats['unique']}</b> unique</span>"
            f"<span><b>{current_stats['short']}</b> very short</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_quality_note(current_reviews)

    ge_button(
        "Clear",
        key="clear_analysis",
        on_click=clear_all,
    )

    if submit:
        name = st.session_state.input_restaurant_name.strip()
        address = st.session_state.input_address.strip()
        reviews = split_reviews(st.session_state.input_review_text)

        if not ready:
            st.error("The trained model artifacts are not available. Add the `gastroeval_artifacts` folder first.")
        elif not name:
            st.error("Enter the restaurant name.")
        elif not address:
            st.error("Enter the restaurant address.")
        elif len(reviews) < 10:
            st.error(f"Add at least 10 reviews before starting the analysis. You currently have {len(reviews)}.")
        else:
            with st.spinner("Analyzing review evidence…"):
                try:
                    result = run_analysis(name, address, reviews)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                else:
                    st.session_state.analysis_restaurant = name
                    st.session_state.analysis_address = address
                    st.session_state.analysis_reviews = reviews
                    st.session_state.result = result
                    st.session_state.view = "results"
                    st.rerun()


# -----------------------------------------------------------------------------
# Results view
# -----------------------------------------------------------------------------
elif st.session_state.view == "results":
    result = st.session_state.get("result")
    if result is None:
        st.session_state.view = "analyze"
        st.rerun()

    restaurant = st.session_state.analysis_restaurant
    address = st.session_state.analysis_address
    reviews = list(st.session_state.analysis_reviews)
    aspect_summary = result["aspect_summary"].copy()
    evidence = result["evidence"].copy()
    score = safe_float(result["restaurant_score"])
    recommendation = str(result["recommendation"])

    st.markdown('<div class="ge-eyebrow">Evaluation result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ge-title" style="font-size:2.55rem;">{esc(restaurant)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="ge-section-copy">{esc(address)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ge-result-head">
            <div>
                <div class="ge-score-line">
                    <div class="ge-score">{score:.1f}</div>
                    <div class="ge-score-unit">/ 100</div>
                </div>
                <div class="ge-meta">
                    <span><b>{len(reviews)}</b> reviews analyzed</span>
                    <span><b>{int(result['evidence_count'])}</b> evidence units</span>
                    <span><b>{safe_float(result['coverage_pct']):.0f}%</b> aspect coverage</span>
                    <span>{esc(result['evidence_quality'])}</span>
                </div>
            </div>
            <div class="ge-recommendation">{esc(recommendation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ge-section-title">Six dimensions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ge-section-copy">Supported aspects contribute to the score. Missing aspects are reported as insufficient evidence rather than automatically penalized.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ge-chart-head"><span class="ge-section-title">Aspect profile</span>'
        '<span class="ge-chart-note">Hover a bar to inspect its score and evidence.</span></div>',
        unsafe_allow_html=True,
    )
    render_aspect_chart(aspect_summary)

    st.markdown('<div class="ge-section"><div class="ge-section-title">Customer evidence</div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ge-section-copy">Representative review contexts behind the evaluation.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Evidence", "All evidence", "Export"])

    with tab1:
        if evidence.empty:
            render_evidence(evidence)
        else:
            filter_left, filter_right = st.columns(2, gap="medium")
            aspect_options = ["All aspects"] + sorted(
                evidence["Aspect"].dropna().astype(str).unique().tolist()
            )
            sentiment_options = ["All sentiments", "Positive", "Neutral", "Negative"]
            with filter_left:
                selected_aspect = st.selectbox(
                    "Aspect",
                    aspect_options,
                    key="result_aspect_filter",
                )
            with filter_right:
                selected_sentiment = st.selectbox(
                    "Sentiment",
                    sentiment_options,
                    key="result_sentiment_filter",
                )

            filtered = evidence.copy()
            if selected_aspect != "All aspects":
                filtered = filtered[filtered["Aspect"].astype(str) == selected_aspect]
            if selected_sentiment != "All sentiments":
                filtered = filtered[
                    filtered["Aspect_Sentiment"].astype(str) == selected_sentiment
                ]

            st.caption(f"Showing {len(filtered)} of {len(evidence)} evidence units")
            render_evidence(filtered, limit=8)

    with tab2:
        if evidence.empty:
            st.info("No evidence available.")
        else:
            display = evidence[
                [
                    "Aspect",
                    "Aspect_Sentiment",
                    "Evaluation_Score",
                    "Sentiment_Confidence",
                    "Aspect_Context",
                ]
            ].copy()
            display.columns = [
                "Aspect",
                "Sentiment",
                "Evaluation score",
                "Confidence",
                "Evidence",
            ]
            ge_dataframe(display, _stretch=True, hide_index=True)

    with tab3:
        st.markdown(
            '<div class="ge-section-copy" style="margin-bottom:14px;">Choose a format. Each file is ready to download immediately.</div>',
            unsafe_allow_html=True,
        )

        csv_bytes = evidence.to_csv(index=False).encode("utf-8")
        html_bytes = build_html_report(result, restaurant, address)

        pdf_bytes = None
        pdf_error = None
        try:
            pdf_bytes = build_pdf_report(result, restaurant, address)
        except Exception as exc:
            pdf_error = str(exc)

        export_left, export_mid, export_right = st.columns(3, gap="medium")

        with export_left:
            st.markdown('<div class="ge-export-type">CSV</div>', unsafe_allow_html=True)
            st.markdown('<div class="ge-export-copy">Structured evidence data for analysis or submission.</div>', unsafe_allow_html=True)
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="GastroEval_evidence.csv",
                mime="text/csv",
                key="download_evidence_csv",
            )

        with export_mid:
            st.markdown('<div class="ge-export-type">HTML</div>', unsafe_allow_html=True)
            st.markdown('<div class="ge-export-copy">Readable report with restaurant score, aspects and evidence.</div>', unsafe_allow_html=True)
            st.download_button(
                "Download HTML",
                data=html_bytes,
                file_name="GastroEval_Report.html",
                mime="text/html",
                key="download_report_html",
            )

        with export_right:
            st.markdown('<div class="ge-export-type">PDF</div>', unsafe_allow_html=True)
            if pdf_bytes is not None:
                st.markdown('<div class="ge-export-copy">Presentation-ready report for sharing or submission.</div>', unsafe_allow_html=True)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="GastroEval_Report.pdf",
                    mime="application/pdf",
                    key="download_report_pdf",
                )
            else:
                st.markdown('<div class="ge-export-copy">PDF export is unavailable in this environment.</div>', unsafe_allow_html=True)
                st.download_button(
                    "PDF unavailable",
                    data=b"",
                    file_name="GastroEval_Report.pdf",
                    mime="application/pdf",
                    key="download_report_pdf_unavailable",
                    disabled=True,
                )
                if pdf_error:
                    st.caption(pdf_error)

    st.markdown("<hr>", unsafe_allow_html=True)
    edit_col, clear_col = st.columns([1.0, .18], gap="small")
    with edit_col:
        ge_button(
            "Edit reviews / start new analysis",
            key="edit_results",
            on_click=goto_view,
            args=("analyze",),
        )
    with clear_col:
        ge_button(
            "Clear",
            key="clear_results",
            _stretch=True,
            on_click=clear_all,
        )


# -----------------------------------------------------------------------------
# Methodology view
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="ge-eyebrow">Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="ge-title">How GastroEval works.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ge-lead">Models, evidence extraction, scoring, and aggregation are shown below.</div>',
        unsafe_allow_html=True,
    )

    blocks = [
        ("01", "Text representation", "Review text is represented with TF-IDF word n-grams from 1 through 5."),
        ("02", "Sentiment", "A supervised Logistic Regression model estimates review sentiment and confidence."),
        ("03", "Aspect evidence", "Domain triggers and linguistic rules identify evidence for Food Quality, Service, Ambience, Value for Money, Hygiene, and Overall Dining Experience."),
        ("04", "Evaluation score", "Model confidence and phrase-level evidence are combined into a continuous 0–100 evaluation score."),
        ("05", "Evidence aggregation", "Multiple contexts from the same review and aspect are collapsed into one review-aspect evidence unit."),
        ("06", "Restaurant score", "Supported aspects are combined using the GastroEval aspect weights. Missing aspects are not automatically penalized."),
    ]

    st.markdown(
        "".join(
            f"""
            <div class="ge-method-row">
                <div class="ge-method-no">{num}</div>
                <div class="ge-method-title">{esc(title)}</div>
                <div class="ge-method-body">{esc(body)}</div>
            </div>
            """
            for num, title, body in blocks
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ge-section"><div class="ge-section-title">Interpretation and limitations</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ge-note">
            GastroEval is a review analytics system rather than ground-truth restaurant grading. Aspect detection combines a restaurant-domain ontology with linguistic rules, while sentiment is learned from the validated TF-IDF + Logistic Regression model. Scores should be interpreted together with the displayed evidence and evidence coverage.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    '<div style="margin-top:3rem;color:#94A3B8;font-size:.66rem;">GastroEval · Explainable restaurant review analytics</div>',
    unsafe_allow_html=True,
)
