from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from essay_pipeline import evaluate_code, evaluate_essay, evaluate_short_answer
import reporting


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="IntelliGrade AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME
# =========================================================
ACCENT = "#0F172A"        # deep slate
ACCENT_BRAND = "#7C3AED"  # violet
ACCENT_SOFT = "#F3E8FF"   # soft lavender
SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"
BORDER = "#E5E7EB"
TEXT = "#0F172A"
TEXT_MUTED = "#64748B"
BG_SOFT = "#F8FAFF"

RUBRIC_PRESETS = {
    "Essay": {
        "Argumentative Essay": "content, coherence, grammar, relevance, structure",
        "Narrative Essay": "content, clarity, grammar, organization, creativity",
        "Analytical Essay": "analysis, evidence, coherence, grammar, structure",
    },
    "Short Answer": {
        "Factual Response": "accuracy, completeness, clarity",
        "Concept Explanation": "understanding, clarity, completeness",
        "Reflective Response": "clarity, reflection, relevance",
    },
    "Code": {
        "Correctness Review": "correctness, readability, efficiency, edge cases",
        "Code Quality": "readability, structure, correctness, style",
        "Algorithmic Solution": "correctness, efficiency, edge cases, explanation",
    },
}

DEFAULT_PROMPTS = {
    "Essay": "Write an argumentative essay about whether schools should limit homework.",
    "Short Answer": "Answer the question clearly and completely in 3 to 5 sentences.",
    "Code": "Write code that solves the problem clearly, correctly, and efficiently.",
}

SUPPORTED_UPLOADS = [
    "txt", "md", "csv", "json", "py", "js", "ts", "java", "c", "cpp", "html", "htm", "docx"
]


# =========================================================
# STYLES
# =========================================================
def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #ffffff 0%, {BG_SOFT} 100%);
            color: {TEXT};
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1320px;
        }}

        /* Overall typography */
        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: {TEXT};
        }}

        .hero-wrap {{
            background: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 26px;
            padding: 22px 24px;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
            margin-bottom: 20px;
        }}

        .hero {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            text-align: left;
            padding: 8px 0;
        }}

        .hero-logo {{
            width: 68px;
            height: 68px;
            min-width: 68px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #EDE9FE, #F8FAFF);
            border: 1px solid {BORDER};
            font-size: 34px;
        }}

        .hero-text {{
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .hero-title {{
            font-size: 2rem;
            font-weight: 900;
            color: {TEXT};
            line-height: 1.1;
        }}

        .hero-subtitle {{
            margin-top: 0.35rem;
            font-size: 0.98rem;
            color: {TEXT_MUTED};
            line-height: 1.5;
            max-width: 920px;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            padding: 7px 12px;
            width: fit-content;
            border-radius: 999px;
            background: {ACCENT_SOFT};
            color: {ACCENT};
            border: 1px solid #E9D5FF;
            font-size: 0.85rem;
            font-weight: 700;
        }}

        .section-title {{
            font-size: 1.08rem;
            font-weight: 900;
            color: {TEXT};
            margin: 0.8rem 0 0.7rem 0;
        }}

        .card {{
            background: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }}

        .card-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {TEXT_MUTED};
            margin-bottom: 0.3rem;
            font-weight: 800;
        }}

        .card-value {{
            font-size: 1.55rem;
            font-weight: 900;
            color: {TEXT} !important;
            line-height: 1.1;
        }}

        .card-note {{
            color: {TEXT_MUTED};
            margin-top: 0.35rem;
            font-size: 0.93rem;
        }}

        .panel {{
            background: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }}

        .preview-box {{
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1rem;
            background: #ffffff;
            color: {TEXT};
            min-height: 150px;
            white-space: pre-wrap;
        }}

        .feedback-box {{
            border-left: 5px solid {ACCENT_BRAND};
            background: #ffffff;
            padding: 16px 18px;
            border-radius: 16px;
            border: 1px solid {BORDER};
            color: {TEXT};
        }}

        .resource-chip {{
            display: inline-block;
            background: {ACCENT_SOFT};
            color: {TEXT};
            padding: 8px 14px;
            border-radius: 999px;
            margin: 4px 8px 4px 0;
            font-size: 0.86rem;
            border: 1px solid #E9D5FF;
            font-weight: 700;
        }}

        .sidebar-brand {{
            background: linear-gradient(180deg, #ffffff, #faf7ff);
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 18px 16px;
            text-align: center;
            margin-bottom: 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .sidebar-logo {{
            width: 58px;
            height: 58px;
            margin: 0 auto 10px auto;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #EDE9FE, #EEF2FF);
            border: 1px solid #DDD6FE;
            font-size: 30px;
        }}

        .sidebar-app {{
            font-size: 1.15rem;
            font-weight: 900;
            color: {TEXT};
            line-height: 1.2;
        }}

        .sidebar-desc {{
            color: {TEXT_MUTED};
            font-size: 0.9rem;
            margin-top: 0.35rem;
            line-height: 1.4;
        }}

        .sidebar-section {{
            padding: 0.35rem 0.15rem 0.55rem 0.15rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid #EEF2FF;
        }}

        .sidebar-title {{
            font-size: 0.9rem;
            font-weight: 900;
            color: {TEXT};
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .small-muted {{
            color: {TEXT_MUTED};
            font-size: 0.92rem;
        }}

        /* Inputs on white background */
        label {{
            color: {TEXT} !important;
            font-weight: 800 !important;
        }}

        input, textarea, select {{
            background: #ffffff !important;
            border: 1px solid #CBD5E1 !important;
            color: {TEXT} !important;
        }}

        input::placeholder, textarea::placeholder {{
            color: #94A3B8 !important;
            opacity: 1 !important;
        }}

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader {{
            background: #ffffff !important;
            color: {TEXT} !important;
        }}

        .stButton > button {{
            background: {ACCENT_BRAND} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            height: 3.1rem !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(124, 58, 237, 0.18);
        }}

        .stButton > button:hover {{
            background: #6D28D9 !important;
            color: white !important;
        }}

        .stMetric label {{
            color: {TEXT_MUTED} !important;
            font-weight: 700 !important;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            color: {TEXT} !important;
            font-weight: 900 !important;
        }}

        .stMarkdown p, .stMarkdown li {{
            color: {TEXT};
        }}

        .footer-note {{
            text-align: center;
            color: {TEXT_MUTED};
            padding: 12px 0 4px 0;
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================
def _preview_text(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if not text:
        return "No submission text yet. Use the sidebar to upload a file or paste text."
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    raw = uploaded_file.getvalue()

    if suffix in {"txt", "md", "csv", "json", "py", "js", "ts", "java", "c", "cpp", "html", "htm"}:
        return raw.decode("utf-8", errors="ignore")

    if suffix == "docx":
        try:
            from docx import Document  # type: ignore[import-not-found]
        except Exception:
            st.sidebar.warning("DOCX upload needs python-docx in this environment.")
            return ""

        document = Document(BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    st.sidebar.warning(f"Unsupported file type: .{suffix}")
    return ""


def _prepare_submission_text(uploaded_file, pasted_text: str) -> str:
    upload_text = _extract_text_from_upload(uploaded_file)
    pasted_text = (pasted_text or "").strip()
    if upload_text and pasted_text:
        return upload_text
    return upload_text or pasted_text


def _parse_similarity_pct(similarity_risk: str) -> int:
    if not similarity_risk:
        return 0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", similarity_risk)
    if m:
        return max(0, min(100, int(round(float(m.group(1))))))
    low = similarity_risk.lower()
    if "high" in low:
        return 85
    if "medium" in low:
        return 55
    if "low" in low:
        return 20
    m2 = re.search(r"(\d+(?:\.\d+)?)", similarity_risk)
    if m2:
        return max(0, min(100, int(round(float(m2.group(1))))))
    return 0


def _score_color(score: float) -> str:
    if score >= 5:
        return SUCCESS
    if score >= 4:
        return ACCENT_BRAND
    if score >= 3:
        return WARNING
    return DANGER


def _render_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            {f'<div class="card-note">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _save_latest_run(
    *,
    assignment_type: str,
    student_name: str,
    department: str,
    rollno: str,
    submission_id: str,
    prompt: str,
    submission_text: str,
    rubric_label: str,
    rubric_text: str,
    evaluation,
    source_name: str,
) -> None:
    st.session_state["latest_run"] = {
        "assignment_type": assignment_type,
        "student_name": student_name,
        "department": department,
        "rollno": rollno,
        "submission_id": submission_id,
        "prompt": prompt,
        "submission_text": submission_text,
        "rubric_label": rubric_label,
        "rubric_text": rubric_text,
        "evaluation": evaluation,
        "source_name": source_name,
    }


def _build_score_gauge(final_score: float, max_score: float = 6.0) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=final_score,
            number={"font": {"color": TEXT, "size": 42}},
            gauge={
                "axis": {"range": [0, max_score], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": ACCENT_BRAND},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, max_score * 0.33], "color": "#FEE2E2"},
                    {"range": [max_score * 0.33, max_score * 0.66], "color": "#FEF3C7"},
                    {"range": [max_score * 0.66, max_score], "color": "#E0E7FF"},
                ],
            },
            title={"text": f"Final Score / {max_score:g}", "font": {"color": TEXT, "size": 16}},
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=20, r=20, t=50, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT),
    )
    return fig


def _build_similarity_gauge(similarity_pct: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=similarity_pct,
            number={"suffix": "%", "font": {"color": TEXT, "size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": WARNING if similarity_pct >= 60 else ACCENT_BRAND},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
            },
            title={"text": "Similarity Risk", "font": {"color": TEXT, "size": 16}},
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=20, r=20, t=50, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT),
    )
    return fig


def _build_rubric_bar_chart(rubric_breakdown: dict) -> go.Figure:
    if not rubric_breakdown:
        fig = go.Figure()
        fig.update_layout(
            height=280,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=TEXT),
            annotations=[
                dict(
                    text="No rubric data available",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=TEXT_MUTED, size=16),
                )
            ],
        )
        return fig

    df = pd.DataFrame(
        {
            "Criterion": list(rubric_breakdown.keys()),
            "Score": list(rubric_breakdown.values()),
        }
    ).sort_values("Score", ascending=True)

    fig = px.bar(
        df,
        x="Score",
        y="Criterion",
        orientation="h",
        text="Score",
        color="Score",
        color_continuous_scale=["#EDE9FE", "#A78BFA", "#7C3AED"],
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        title={"text": "Rubric Breakdown", "font": {"color": TEXT, "size": 16}},
        height=320,
        margin=dict(l=10, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT),
        xaxis=dict(gridcolor="#E2E8F0", zeroline=False, color=TEXT),
        yaxis=dict(color=TEXT),
        coloraxis_showscale=False,
    )
    return fig


def _build_rubric_radar_chart(rubric_breakdown: dict) -> go.Figure:
    if not rubric_breakdown or len(rubric_breakdown) < 2:
        fig = go.Figure()
        fig.update_layout(
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=TEXT),
            annotations=[
                dict(
                    text="Need at least two rubric points for radar view",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=TEXT_MUTED, size=16),
                )
            ],
        )
        return fig

    labels = list(rubric_breakdown.keys())
    values = list(rubric_breakdown.values())
    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line=dict(color=ACCENT_BRAND, width=2),
            fillcolor="rgba(124, 58, 237, 0.20)",
            name="Rubric",
        )
    )
    fig.update_layout(
        title={"text": "Rubric Shape", "font": {"color": TEXT, "size": 16}},
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        polar=dict(
            radialaxis=dict(
                visible=True,
                color=TEXT,
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(color=TEXT),
            bgcolor="white",
        ),
        showlegend=False,
        font=dict(color=TEXT),
    )
    return fig


def _build_resources_chart(evaluation) -> go.Figure:
    resources = getattr(evaluation, "resources", []) or []
    if not resources:
        fig = go.Figure()
        fig.update_layout(
            height=260,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=TEXT),
            annotations=[
                dict(
                    text="No resources available",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=TEXT_MUTED, size=16),
                )
            ],
        )
        return fig

    df = pd.DataFrame({"Resource": resources, "Count": [1] * len(resources)})
    fig = px.bar(
        df,
        x="Resource",
        y="Count",
        text="Count",
        template="plotly_white",
        color_discrete_sequence=["#7C3AED"],
    )
    fig.update_layout(
        title={"text": "Recommended Resources", "font": {"color": TEXT, "size": 16}},
        height=280,
        margin=dict(l=10, r=10, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, tickangle=-25),
        yaxis=dict(color=TEXT, visible=False),
    )
    return fig


# =========================================================
# APP
# =========================================================
_inject_styles()

# ---------------------------------------------------------
# SIDEBAR BRANDING
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🧠</div>
            <div class="sidebar-app">IntelliGrade AI</div>
            <div class="sidebar-desc">
                Automated assignment evaluation with rubric scoring, feedback, and report export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-section">
            <div class="sidebar-title">Quick Start</div>
            <div class="small-muted">1. Enter student details<br>2. Paste or upload submission<br>3. Click evaluate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Open evaluator", use_container_width=True):
        st.session_state["show_evaluator"] = True

    st.markdown(
        f"""
        <div class="sidebar-section">
            <div class="sidebar-title">Theme</div>
            <div class="small-muted">
                White background<br>
                Indigo / Violet accents<br>
                Clean dashboard layout
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.get("show_evaluator", False):
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero">
                <div class="hero-logo">🧠</div>
                <div class="hero-text">
                    <div class="hero-title">IntelliGrade AI</div>
                    <div class="hero-subtitle">
                        Automated rubric-based assignment evaluation with semantic scoring,
                        personalized feedback, similarity risk, and downloadable reports.
                    </div>
                    <div class="hero-badge">✨ Built for fast, transparent evaluation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Click 'Open evaluator' in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero">
            <div class="hero-logo">🧠</div>
            <div class="hero-text">
                <div class="hero-title">IntelliGrade AI</div>
                <div class="hero-subtitle">
                    Automated rubric-based assignment evaluation with semantic scoring,
                    personalized feedback, similarity risk, and downloadable reports.
                </div>
                <div class="hero-badge">🚀 Professional assessment dashboard for hackathons and demos</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# INPUT PANEL
# ---------------------------------------------------------
st.markdown('<div class="section-title">Submission Setup</div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Student Details")

        c1, c2 = st.columns(2)
        with c1:
            student_name = st.text_input("Student name", key="student_name")
            rollno = st.text_input("Roll no.", key="rollno")
        with c2:
            division = st.text_input("Division", key="division")
            submission_id = st.text_input("Submission ID", key="submission_id")

        st.markdown("#### Assignment Details")
        rubric_type = st.selectbox("Assignment type", list(RUBRIC_PRESETS.keys()), index=0)
        rubric_choice = st.selectbox(
            "Rubric preset",
            list(RUBRIC_PRESETS[rubric_type].keys()),
            index=0,
        )

        topic = st.text_input(
            "Assessment topic / prompt",
            value=DEFAULT_PROMPTS[rubric_type],
            key="topic",
        )

        rubric_text = st.text_area(
            "Rubric or evaluation guidance",
            value=RUBRIC_PRESETS[rubric_type][rubric_choice],
            height=115,
            key="rubric_text",
        )
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Submission Upload")
    uploaded_file = st.file_uploader("Upload student submission", type=SUPPORTED_UPLOADS)
    pasted_text = st.text_area("Or paste student submission text here", height=290)
    st.markdown("</div>", unsafe_allow_html=True)

submission_text = _prepare_submission_text(uploaded_file, pasted_text)
source_name = uploaded_file.name if uploaded_file is not None else "Pasted text"

st.markdown("")
evaluate_clicked = st.button("Submit evaluation", type="primary", use_container_width=True)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
if evaluate_clicked:
    if not submission_text.strip():
        st.error("Upload a file or paste submission text before evaluating.")
    elif not student_name.strip():
        st.error("Enter the student name.")
    else:
        with st.spinner("Running checkpoint-backed evaluation..."):
            if rubric_type == "Essay":
                evaluation = evaluate_essay(
                    text=submission_text,
                    prompt=topic,
                    rubric=rubric_text,
                    reference_texts=[],
                )
            elif rubric_type == "Short Answer":
                evaluation = evaluate_short_answer(
                    text=submission_text,
                    reference_texts=[],
                )
            elif rubric_type == "Code":
                evaluation = evaluate_code(
                    text=submission_text,
                    reference_texts=[],
                )
            else:
                st.error(f"Unsupported assignment type: {rubric_type}")
                evaluation = None

        if evaluation is not None:
            _save_latest_run(
                assignment_type=rubric_type,
                student_name=student_name,
                department=division,
                rollno=rollno,
                submission_id=submission_id,
                prompt=topic,
                submission_text=submission_text,
                rubric_label=rubric_choice,
                rubric_text=rubric_text,
                evaluation=evaluation,
                source_name=source_name,
            )
            st.success("Evaluation complete. Scroll down to see the dashboard.")

latest_run = st.session_state.get("latest_run")

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
if latest_run:
    evaluation = latest_run["evaluation"]
    similarity_pct = _parse_similarity_pct(getattr(evaluation, "similarity_risk", ""))

    st.markdown("---")
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Final Score", f"{evaluation.final_score}/6")
    k2.metric("Confidence", f"{evaluation.confidence:.2f}")
    k3.metric("Similarity Risk", getattr(evaluation, "similarity_risk", "N/A"))
    k4.metric("Model Status", getattr(evaluation, "model_status", "N/A"))

    # Visuals
    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        st.plotly_chart(_build_score_gauge(float(evaluation.final_score), 6.0), use_container_width=True)
    with c_right:
        st.plotly_chart(_build_similarity_gauge(similarity_pct), use_container_width=True)

    st.markdown('<div class="section-title">Rubric Visualisation</div>', unsafe_allow_html=True)
    r_left, r_right = st.columns(2, gap="large")
    with r_left:
        st.plotly_chart(_build_rubric_bar_chart(getattr(evaluation, "rubric_breakdown", {})), use_container_width=True)
    with r_right:
        st.plotly_chart(_build_rubric_radar_chart(getattr(evaluation, "rubric_breakdown", {})), use_container_width=True)

    # Student card
    st.markdown('<div class="section-title">Student Details</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Student", latest_run["student_name"] or "-")
    s2.metric("Roll No.", latest_run["rollno"] or "-")
    s3.metric("Division", latest_run["department"] or "-")
    s4.metric("Submission ID", latest_run["submission_id"] or "-")

    # Analysis tabs
    st.markdown('<div class="section-title">Analysis & Feedback</div>', unsafe_allow_html=True)
    tab_analysis, tab_feedback, tab_resources = st.tabs(["📈 Analysis", "💬 Feedback", "📚 Resources"])

    with tab_analysis:
        a_left, a_right = st.columns([1.15, 0.85], gap="large")

        with a_left:
            st.markdown("#### Prompt")
            st.info(latest_run["prompt"] or "No prompt provided.")

            st.markdown("#### Submission Preview")
            st.markdown(
                f"""
                <div class="preview-box">{_preview_text(latest_run["submission_text"], limit=1800)}</div>
                """,
                unsafe_allow_html=True,
            )

        with a_right:
            st.markdown("#### Summary Cards")
            _render_card("Overall status", getattr(evaluation, "model_status", "N/A"), "Based on score and similarity risk.")
            _render_card("Final score", f"{evaluation.final_score}/6", "Normalized evaluation output.")
            _render_card("Similarity risk", getattr(evaluation, "similarity_risk", "N/A"), "Lower is better.")
            _render_card("Confidence", f"{evaluation.confidence:.2f}", "Model certainty.")

        if getattr(evaluation, "rubric_breakdown", None):
            st.markdown("#### Rubric Breakdown Table")
            rubric_df = pd.DataFrame(
                {
                    "Criterion": list(evaluation.rubric_breakdown.keys()),
                    "Score": list(evaluation.rubric_breakdown.values()),
                }
            )
            st.dataframe(rubric_df, use_container_width=True, hide_index=True)

    with tab_feedback:
        st.markdown("#### Personalized Feedback")
        st.markdown(
            f"""
            <div class="feedback-box">
                <b>Feedback:</b> {evaluation.feedback}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("#### Strengths")
            for item in getattr(evaluation, "strengths", []):
                st.success(item)

        with col_w:
            st.markdown("#### Areas to Improve")
            for item in getattr(evaluation, "weaknesses", []):
                st.warning(item)

    with tab_resources:
        st.markdown("#### Recommended Resources")
        resources = getattr(evaluation, "resources", []) or []
        for item in resources:
            if isinstance(item, dict):
                title = item.get("title", "Resource")
                url = item.get("url", "#")
                st.markdown(
                    f"<span class='resource-chip'><a href='{url}' target='_blank' style='text-decoration:none;color:inherit;'>{title}</a></span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<span class='resource-chip'>{item}</span>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Resource Selection Logic")
        detected = getattr(evaluation, "weaknesses", []) or []
        st.write(f"Detected weaknesses: {', '.join(detected) if detected else 'None detected'}")
        if resources:
            st.write("Resources should map directly to the detected weakness categories.")
        else:
            st.write("No resources were returned by the backend for this submission.")

    # Export
    st.markdown("---")
    st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        try:
            report_bytes = reporting.build_docx_report(
                submission_id=latest_run["submission_id"],
                student_name=latest_run["student_name"],
                prompt=latest_run["prompt"],
                essay_text=latest_run["submission_text"],
                evaluation=evaluation,
            )
            st.download_button(
                "Download report (DOCX)",
                data=report_bytes,
                file_name=reporting.docx_filename(
                    latest_run["submission_id"] or latest_run["student_name"] or "submission"
                ),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception:
            st.warning("DOCX export is unavailable. Install python-docx to enable it.")

    with c2:
        try:
            pdf_bytes = reporting.build_pdf_report(
                submission_id=latest_run["submission_id"],
                student_name=latest_run["student_name"],
                prompt=latest_run["prompt"],
                essay_text=latest_run["submission_text"],
                evaluation=evaluation,
            )
            st.download_button(
                "Download report (PDF)",
                data=pdf_bytes,
                file_name=reporting.pdf_filename(
                    latest_run["submission_id"] or latest_run["student_name"] or "submission"
                ),
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.info("PDF export is unavailable. Install reportlab to enable it.")

    st.markdown(
        f"""
        <div class="footer-note">
            Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} • IntelliGrade AI
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown("---")
    st.markdown("### Dashboard Preview")
    p1, p2, p3 = st.columns(3)
    with p1:
        _render_card("Final score", "—", "Will appear after evaluation.")
    with p2:
        _render_card("Similarity risk", "—", "Will appear after evaluation.")
    with p3:
        _render_card("Confidence", "—", "Will appear after evaluation.")

    st.caption("Fill the form and click Submit evaluation to generate the dashboard.")

st.markdown("---")
st.caption("Flow: input → model evaluation → rubric analysis → visual dashboard → feedback → export.")