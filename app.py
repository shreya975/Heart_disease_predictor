"""
=================================================================================
AI HEART DISEASE PREDICTION SYSTEM
Clinical Intelligence Powered by Machine Learning
Single-file premium Streamlit application.
=================================================================================
"""

import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# =================================================================================
# PAGE CONFIG
# =================================================================================
st.set_page_config(
    page_title="AI Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose",
]

MODEL_PATH = "heart_disease_model.pkl"
DATA_PATH = "framingham.csv"


# =================================================================================
# CUSTOM CSS — GLASSMORPHISM / NEUMORPHISM / GRADIENTS / ANIMATIONS
# =================================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root{
        --cyan:#00E5FF;
        --purple:#7C3AED;
        --deep-purple:#5B21B6;
        --sky:#0EA5E9;
        --dark:#111827;
        --darker:#1F2937;
        --card-radius:26px;
        --glass-bg: rgba(255,255,255,0.05);
        --glass-border: rgba(255,255,255,0.12);
    }

    /* ---------- HIDE STREAMLIT BRANDING ---------- */
    #MainMenu {visibility:hidden;}
    header {visibility:hidden;}
    footer {visibility:hidden;}
    div[data-testid="stToolbar"] {visibility:hidden; height:0;}
    div[data-testid="stDecoration"] {visibility:hidden; height:0;}
    div[data-testid="stStatusWidget"] {visibility:hidden;}
    a[href*="streamlit.io"] {display:none !important;}
    .viewerBadge_container__1QSob {display:none !important;}

    /* ---------- GLOBAL ---------- */
    html, body, [class*="css"]  {
        font-family:'Manrope', sans-serif;
    }
    h1,h2,h3,h4, .hero-title {
        font-family:'Space Grotesk', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #0b0f19, #111827, #1a1035, #0b0f19);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        color: #E5E7EB;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* floating background orbs */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.35;
        z-index: 0;
        animation: floatOrb 12s ease-in-out infinite;
        pointer-events: none;
    }
    .orb1 {width:380px; height:380px; background:var(--cyan); top:-100px; left:-100px;}
    .orb2 {width:420px; height:420px; background:var(--purple); bottom:-140px; right:-120px; animation-delay: 3s;}
    .orb3 {width:300px; height:300px; background:var(--sky); top:40%; right:10%; animation-delay: 6s;}

    @keyframes floatOrb {
        0%,100% {transform: translateY(0px) translateX(0px);}
        50% {transform: translateY(-40px) translateX(30px);}
    }

    /* ---------- CUSTOM SCROLLBAR ---------- */
    ::-webkit-scrollbar {width: 10px; height:10px;}
    ::-webkit-scrollbar-track {background: #0b0f19;}
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--cyan), var(--purple));
        border-radius: 10px;
    }

    /* ---------- GLASS CARD ---------- */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid var(--glass-border);
        border-radius: var(--card-radius);
        padding: 26px 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: all 0.35s ease;
        animation: fadeInUp 0.7s ease;
        margin-bottom: 20px;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 44px rgba(124,58,237,0.25), 0 0 0 1px rgba(0,229,255,0.15);
        border-color: rgba(0,229,255,0.35);
    }

    .neu-card {
        background: linear-gradient(145deg, #161d2e, #0d121e);
        border-radius: var(--card-radius);
        box-shadow:  10px 10px 24px #0a0e18, -10px -10px 24px #1a2338;
        padding: 22px 24px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.04);
    }

    @keyframes fadeInUp {
        from {opacity:0; transform: translateY(24px);}
        to {opacity:1; transform: translateY(0);}
    }

    /* ---------- HERO ---------- */
    .hero-wrap {text-align:center; padding: 18px 0 6px 0; position:relative; z-index:1;}
    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #7C3AED 50%, #0EA5E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: shine 5s linear infinite;
        margin-bottom: 6px;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #9CA3AF;
        font-weight: 500;
        margin-bottom: 18px;
        letter-spacing: 0.3px;
    }
    .heartbeat {
        display:inline-block;
        animation: heartbeat 1.4s ease-in-out infinite;
    }
    @keyframes heartbeat {
        0%,100% {transform: scale(1);}
        20% {transform: scale(1.25);}
        40% {transform: scale(1);}
        60% {transform: scale(1.2);}
        80% {transform: scale(1);}
    }

    .badge-row {display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom: 22px;}
    .badge-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(0,229,255,0.35);
        color: #E0F7FA;
        padding: 7px 18px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .badge-pill:hover {
        background: rgba(0,229,255,0.15);
        transform: translateY(-3px);
    }

    /* ---------- SECTION HEADERS ---------- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 6px 0 14px 0;
        padding-left: 12px;
        border-left: 4px solid var(--cyan);
    }

    /* ---------- BUTTONS ---------- */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00E5FF, #7C3AED);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 14px 0;
        border: none;
        border-radius: 18px;
        box-shadow: 0 0 18px rgba(0,229,255,0.45), 0 0 36px rgba(124,58,237,0.35);
        transition: all 0.3s ease;
        animation: pulseGlow 2.4s ease-in-out infinite;
        letter-spacing: 0.4px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 28px rgba(0,229,255,0.7), 0 0 56px rgba(124,58,237,0.55);
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }
    @keyframes pulseGlow {
        0%,100% {box-shadow: 0 0 18px rgba(0,229,255,0.45), 0 0 36px rgba(124,58,237,0.35);}
        50% {box-shadow: 0 0 26px rgba(0,229,255,0.75), 0 0 50px rgba(124,58,237,0.6);}
    }

    /* ---------- RESULT CARDS ---------- */
    .result-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(6,95,70,0.25));
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 28px;
        padding: 32px;
        text-align:center;
        animation: fadeInUp 0.6s ease;
        box-shadow: 0 0 40px rgba(16,185,129,0.15);
    }
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(127,29,29,0.28));
        border: 1px solid rgba(239,68,68,0.55);
        border-radius: 28px;
        padding: 32px;
        text-align:center;
        animation: fadeInUp 0.6s ease, shake 0.6s ease;
        box-shadow: 0 0 40px rgba(239,68,68,0.18);
    }
    @keyframes shake {
        0%,100% {transform: translateX(0);}
        25% {transform: translateX(-4px);}
        75% {transform: translateX(4px);}
    }
    .big-icon {font-size: 3.6rem; margin-bottom: 6px;}
    .risk-word {font-size: 1.8rem; font-weight:800; margin-bottom:4px;}

    .warn-box {
        background: rgba(239,68,68,0.12);
        border: 1px dashed rgba(239,68,68,0.6);
        border-radius: 18px;
        padding: 16px 18px;
        margin-top: 14px;
        color: #FCA5A5;
        font-weight: 600;
        font-size: 0.92rem;
    }

    /* ---------- STATUS BADGES ---------- */
    .status-badge {
        display:inline-block;
        padding: 5px 16px;
        border-radius: 999px;
        font-weight:700;
        font-size: 0.8rem;
    }
    .status-good {background: rgba(16,185,129,0.18); color:#34D399; border:1px solid rgba(16,185,129,0.5);}
    .status-bad {background: rgba(239,68,68,0.18); color:#F87171; border:1px solid rgba(239,68,68,0.5);}

    /* ---------- MISC WIDGET TWEAKS ---------- */
    .stSlider [data-baseweb="slider"] {padding-top: 4px;}
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 14px !important;
        border-color: rgba(255,255,255,0.15) !important;
    }
    .stRadio > div {gap: 6px;}
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04);
        border-radius: 14px !important;
    }

    /* metric mini cards */
    .mini-metric {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 14px 16px;
        text-align:center;
        transition: all 0.3s ease;
    }
    .mini-metric:hover {transform: translateY(-4px); border-color: rgba(0,229,255,0.4);}
    .mini-metric .val {font-size:1.5rem; font-weight:800; color:#fff;}
    .mini-metric .lbl {font-size:0.75rem; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;}

    .footer-text {
        text-align:center;
        color:#6B7280;
        font-size:0.85rem;
        padding: 26px 0 10px 0;
    }
    </style>

    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
    """, unsafe_allow_html=True)


# =================================================================================
# DATA + MODEL
# =================================================================================
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    Loads a pre-trained model from disk (pickle) if available.
    Otherwise trains a fallback RandomForest pipeline on framingham.csv
    so the app is fully functional out of the box.
    Swap in your own trained model by saving it to `heart_disease_model.pkl`.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        metrics = {"accuracy": None, "auc": None, "source": "Loaded from heart_disease_model.pkl"}
        return model, metrics
    except Exception:
        pass

    df = load_data()
    if df is None:
        return None, {"accuracy": None, "auc": None, "source": "No model or dataset found"}

    x = df.drop(columns="TenYearCHD")
    y = df["TenYearCHD"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
        "source": "Trained locally on framingham.csv (fallback model)",
    }
    return model, metrics


def predict(data: dict):
    """
    Central prediction function.
    `data` is a dict of raw patient inputs keyed by FEATURES.
    Returns a dict with probability, prediction label, and confidence.
    Replace / extend this if your trained model needs different preprocessing.
    """
    model, _ = load_or_train_model()
    row = pd.DataFrame([{k: data.get(k) for k in FEATURES}])

    if model is None:
        prob = 0.0
    else:
        prob = float(model.predict_proba(row)[0][1])

    label = 1 if prob >= 0.5 else 0
    confidence = max(prob, 1 - prob) * 100
    return {"probability": prob, "label": label, "confidence": confidence}


def compute_health_score(data: dict, risk_prob: float) -> int:
    """Rule-of-thumb wellness score (0-100), independent of the ML risk output."""
    score = 100

    bmi = data["BMI"]
    if bmi < 18.5 or bmi > 30:
        score -= 12
    elif bmi > 25:
        score -= 6

    if data["sysBP"] > 140 or data["diaBP"] > 90:
        score -= 14
    elif data["sysBP"] > 130:
        score -= 7

    if data["totChol"] > 240:
        score -= 12
    elif data["totChol"] > 200:
        score -= 6

    if data["glucose"] > 125:
        score -= 10
    elif data["glucose"] > 100:
        score -= 5

    if data["currentSmoker"] == 1:
        score -= 10 + min(data["cigsPerDay"], 20) * 0.3

    if data["diabetes"] == 1:
        score -= 10
    if data["prevalentHyp"] == 1:
        score -= 6
    if data["prevalentStroke"] == 1:
        score -= 12
    if data["BPMeds"] == 1:
        score -= 4

    score -= risk_prob * 20

    return int(max(5, min(100, round(score))))


# =================================================================================
# CHART HELPERS
# =================================================================================
def gauge_chart(value, title, max_value=100, color="#00E5FF", suffix="%"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 34, "color": "#F9FAFB"}},
        title={"text": title, "font": {"size": 15, "color": "#9CA3AF"}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "#4B5563"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_value * 0.4], "color": "rgba(16,185,129,0.18)"},
                {"range": [max_value * 0.4, max_value * 0.7], "color": "rgba(245,158,11,0.18)"},
                {"range": [max_value * 0.7, max_value], "color": "rgba(239,68,68,0.18)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=220,
        font={"color": "#E5E7EB"},
    )
    return fig


def radar_chart(data: dict):
    categories = ["BMI", "Sys BP", "Dia BP", "Cholesterol", "Glucose", "Heart Rate"]
    patient_vals = [
        min(data["BMI"] / 40 * 100, 100),
        min(data["sysBP"] / 200 * 100, 100),
        min(data["diaBP"] / 120 * 100, 100),
        min(data["totChol"] / 320 * 100, 100),
        min(data["glucose"] / 200 * 100, 100),
        min(data["heartRate"] / 120 * 100, 100),
    ]
    healthy_ref = [55, 55, 55, 50, 45, 50]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=healthy_ref, theta=categories, fill="toself", name="Healthy Reference",
        line=dict(color="#0EA5E9"), fillcolor="rgba(14,165,233,0.12)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_vals, theta=categories, fill="toself", name="Patient Profile",
        line=dict(color="#00E5FF"), fillcolor="rgba(0,229,255,0.22)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#6B7280", gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(color="#9CA3AF", gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#E5E7EB")),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=20, b=20),
        height=340,
        font={"color": "#E5E7EB"},
    )
    return fig


# =================================================================================
# SIDEBAR
# =================================================================================
def render_sidebar(metrics):
    with st.sidebar:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2rem;">🫀</div>
            <div style="font-weight:800; font-size:1.05rem; color:#fff;">Cardio AI</div>
            <div style="font-size:0.78rem; color:#9CA3AF;">Clinical Prediction Suite</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📌 Navigate</div>', unsafe_allow_html=True)
        st.radio(
            "nav", ["🏠 Home", "🩺 Prediction", "📊 Analytics", "❓ FAQ"],
            label_visibility="collapsed", key="nav_radio"
        )

        st.markdown('<div class="section-title">🧠 Model Information</div>', unsafe_allow_html=True)
        acc_txt = f"{metrics['accuracy']*100:.1f}%" if metrics.get("accuracy") else "—"
        auc_txt = f"{metrics['auc']*100:.1f}%" if metrics.get("auc") else "—"
        st.markdown(f"""
        <div class="neu-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#9CA3AF; font-size:0.85rem;">Algorithm</span>
                <span style="color:#fff; font-weight:700; font-size:0.85rem;">Random Forest</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#9CA3AF; font-size:0.85rem;">Accuracy</span>
                <span style="color:#00E5FF; font-weight:700; font-size:0.85rem;">{acc_txt}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#9CA3AF; font-size:0.85rem;">ROC-AUC</span>
                <span style="color:#7C3AED; font-weight:700; font-size:0.85rem;">{auc_txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📁 About Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="neu-card" style="font-size:0.85rem; color:#D1D5DB; line-height:1.5;">
            Framingham Heart Study — a long-running cardiovascular cohort dataset
            with demographic, behavioral, and clinical features used to estimate
            <b>10-year Coronary Heart Disease (CHD) risk</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔮 About Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="neu-card" style="font-size:0.85rem; color:#D1D5DB; line-height:1.5;">
            The model outputs a probability of developing CHD within 10 years,
            alongside a rule-based wellness Health Score. This tool is for
            educational purposes and is not a medical diagnosis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer-text">Made with ❤️ + Python</div>
        """, unsafe_allow_html=True)


# =================================================================================
# HERO
# =================================================================================
def render_hero():
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">
            <span class="heartbeat">❤️</span> AI Heart Disease Prediction System
        </div>
        <div class="hero-sub">Clinical Intelligence Powered by Machine Learning</div>
        <div class="badge-row">
            <div class="badge-pill">✔ AI Powered</div>
            <div class="badge-pill">✔ ML Prediction</div>
            <div class="badge-pill">✔ Real-Time Analysis</div>
            <div class="badge-pill">✔ Clinical Grade</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =================================================================================
# INPUT FORM
# =================================================================================
def render_input_form():
    inputs = {}

    st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["age"] = st.slider("Age", 20, 90, 45)
        with c2:
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            inputs["male"] = 1 if sex == "Male" else 0
        with c3:
            edu_label = st.selectbox(
                "Education Level",
                ["Some High School", "High School / GED", "Some College", "College or Above"],
            )
            inputs["education"] = ["Some High School", "High School / GED", "Some College", "College or Above"].index(edu_label) + 1
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["sysBP"] = st.slider("Systolic BP (mmHg)", 80, 260, 120)
            inputs["diaBP"] = st.slider("Diastolic BP (mmHg)", 50, 160, 80)
        with c2:
            inputs["totChol"] = st.slider("Total Cholesterol (mg/dL)", 100, 600, 200)
            inputs["BMI"] = st.slider("BMI", 15.0, 50.0, 24.5, step=0.1)
        with c3:
            inputs["heartRate"] = st.slider("Heart Rate (bpm)", 40, 140, 75)
            inputs["glucose"] = st.slider("Glucose (mg/dL)", 40, 400, 90)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚬 Lifestyle</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            smoker = st.radio("Current Smoker", ["No", "Yes"], horizontal=True)
            inputs["currentSmoker"] = 1 if smoker == "Yes" else 0
        with c2:
            inputs["cigsPerDay"] = st.slider(
                "Cigarettes Per Day", 0, 60, 0, disabled=(inputs["currentSmoker"] == 0)
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧬 Medical History</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v = st.radio("Diabetes", ["No", "Yes"], horizontal=True, key="diab")
            inputs["diabetes"] = 1 if v == "Yes" else 0
        with c2:
            v = st.radio("BP Medication", ["No", "Yes"], horizontal=True, key="bpmeds")
            inputs["BPMeds"] = 1 if v == "Yes" else 0
        with c3:
            v = st.radio("Prior Stroke", ["No", "Yes"], horizontal=True, key="stroke")
            inputs["prevalentStroke"] = 1 if v == "Yes" else 0
        with c4:
            v = st.radio("Hypertension", ["No", "Yes"], horizontal=True, key="hyp")
            inputs["prevalentHyp"] = 1 if v == "Yes" else 0
        st.markdown('</div>', unsafe_allow_html=True)

    return inputs


# =================================================================================
# LIVE ANALYSIS PANEL (right column)
# =================================================================================
def render_live_panel(result, health_score):
    st.markdown('<div class="section-title">📡 Live AI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if result is None:
        st.markdown("""
        <div style="text-align:center; color:#9CA3AF; padding: 10px 0;">
            <div style="font-size:2rem;">🤖</div>
            Run a prediction to see live AI analysis.
        </div>
        """, unsafe_allow_html=True)
    else:
        status_class = "status-bad" if result["label"] == 1 else "status-good"
        status_text = "High Risk" if result["label"] == 1 else "Low Risk"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="val">{result['probability']*100:.1f}%</div>
                <div class="lbl">Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="val">{health_score}</div>
                <div class="lbl">Health Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="val">{result['confidence']:.1f}%</div>
                <div class="lbl">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="mini-metric">
                <span class="status-badge {status_class}">{status_text}</span>
                <div class="lbl" style="margin-top:8px;">Status</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700; color:#F3F4F6; margin-bottom:8px;">💡 Recommendations</div>', unsafe_allow_html=True)

        tips = []
        if result["label"] == 1:
            tips = [
                "Schedule a cardiology consultation soon",
                "Monitor blood pressure & cholesterol regularly",
                "Reduce sodium and saturated fat intake",
            ]
        else:
            tips = [
                "Maintain current healthy lifestyle habits",
                "Continue regular annual checkups",
                "Keep up consistent physical activity",
            ]
        for t in tips:
            st.markdown(f"""
            <div class="neu-card" style="padding:12px 16px; font-size:0.85rem; margin-bottom:8px;">
                ✅ {t}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =================================================================================
# RESULT SECTION
# =================================================================================
def render_result(result, inputs, health_score):
    prob_pct = result["probability"] * 100

    if result["label"] == 0:
        st.markdown(f"""
        <div class="result-low">
            <div class="big-icon">🛡️</div>
            <div class="risk-word" style="color:#34D399;">Low Risk Detected</div>
            <div style="color:#A7F3D0; font-size:0.95rem;">
                Estimated 10-year CHD risk probability: <b>{prob_pct:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-high">
            <div class="big-icon">⚠️</div>
            <div class="risk-word" style="color:#F87171;">High Risk Detected</div>
            <div style="color:#FCA5A5; font-size:0.95rem;">
                Estimated 10-year CHD risk probability: <b>{prob_pct:.1f}%</b>
            </div>
            <div class="warn-box">
                🚨 This result indicates elevated cardiovascular risk. Please consult a
                qualified healthcare professional for a full clinical evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(prob_pct, "Risk Meter", color="#EF4444" if result["label"] else "#10B981"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(health_score, "Health Score", color="#00E5FF"), use_container_width=True)
    with g3:
        st.plotly_chart(gauge_chart(inputs["BMI"], "BMI Indicator", max_value=50, suffix="", color="#7C3AED"), use_container_width=True)

    st.markdown('<div class="section-title">📊 Feature Comparison (Patient vs Healthy Reference)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(radar_chart(inputs), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =================================================================================
# BOTTOM SECTION
# =================================================================================
def render_bottom_section(df, metrics):
    st.markdown('<div class="section-title">🌿 Health Tips & Insights</div>', unsafe_allow_html=True)

    tabs = st.tabs(["💡 Health Tips", "❓ FAQs", "🧠 About Model", "📈 Dataset Stats"])

    with tabs[0]:
        tip_cols = st.columns(3)
        tips = [
            ("🥗", "Eat Heart-Healthy", "Favor vegetables, whole grains, and lean proteins over processed food."),
            ("🏃", "Stay Active", "Aim for at least 150 minutes of moderate exercise per week."),
            ("🚭", "Avoid Smoking", "Quitting smoking rapidly lowers cardiovascular risk over time."),
            ("😴", "Sleep Well", "7-9 hours of quality sleep supports healthy blood pressure."),
            ("🧘", "Manage Stress", "Chronic stress contributes to hypertension and heart strain."),
            ("🩺", "Regular Checkups", "Routine screening catches risk factors before symptoms appear."),
        ]
        for i, (icon, title, desc) in enumerate(tips):
            with tip_cols[i % 3]:
                st.markdown(f"""
                <div class="neu-card" style="min-height:150px;">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="font-weight:700; color:#fff; margin:6px 0;">{title}</div>
                    <div style="font-size:0.82rem; color:#9CA3AF;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[1]:
        faqs = [
            ("Is this a medical diagnosis?", "No. This tool provides an educational risk estimate only and does not replace professional medical advice."),
            ("What data powers this model?", "The Framingham Heart Study dataset — a well-known cardiovascular research cohort."),
            ("How is the risk calculated?", "A machine learning model estimates the probability of developing CHD within 10 years based on your inputs."),
            ("What is the Health Score?", "A separate rule-based wellness indicator (0-100) reflecting how your metrics compare to healthy ranges."),
        ]
        for q, a in faqs:
            with st.expander(q):
                st.write(a)

    with tabs[2]:
        acc_txt = f"{metrics['accuracy']*100:.2f}%" if metrics.get("accuracy") else "N/A"
        auc_txt = f"{metrics['auc']*100:.2f}%" if metrics.get("auc") else "N/A"
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="mini-metric"><div class="val">{acc_txt}</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="mini-metric"><div class="val">{auc_txt}</div><div class="lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="mini-metric"><div class="val">15</div><div class="lbl">Input Features</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="neu-card" style="margin-top:14px; font-size:0.85rem; color:#D1D5DB;">
            <b>Source:</b> {metrics.get('source','N/A')}<br>
            <b>Algorithm:</b> Random Forest Classifier (median-imputed, class-balanced)<br>
            <b>Target:</b> TenYearCHD — 10-year Coronary Heart Disease risk
        </div>
        """, unsafe_allow_html=True)

    with tabs[3]:
        if df is not None:
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(f'<div class="mini-metric"><div class="val">{len(df):,}</div><div class="lbl">Patients</div></div>', unsafe_allow_html=True)
            with d2:
                st.markdown(f'<div class="mini-metric"><div class="val">{df["TenYearCHD"].mean()*100:.1f}%</div><div class="lbl">CHD Prevalence</div></div>', unsafe_allow_html=True)
            with d3:
                st.markdown(f'<div class="mini-metric"><div class="val">{df["age"].mean():.0f}</div><div class="lbl">Avg. Age</div></div>', unsafe_allow_html=True)
            with d4:
                st.markdown(f'<div class="mini-metric"><div class="val">{(df["currentSmoker"].mean()*100):.0f}%</div><div class="lbl">Smokers</div></div>', unsafe_allow_html=True)
        else:
            st.info("Dataset not found in the deployment directory.")

    st.markdown("""
    <div class="footer-text">
        © 2026 Cardio AI · Built for educational purposes · Not a substitute for professional medical advice<br>
        Made with ❤️ + Python
    </div>
    """, unsafe_allow_html=True)


# =================================================================================
# MAIN APP
# =================================================================================
def main():
    inject_custom_css()

    model, metrics = load_or_train_model()
    df = load_data()

    render_sidebar(metrics)
    render_hero()

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.inputs = None
        st.session_state.health_score = None

    left, right = st.columns([2.1, 1])

    with left:
        inputs = render_input_form()

        predict_clicked = st.button("🔮  Predict Heart Disease Risk")

        if predict_clicked:
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                bar = st.progress(0, text="🧠 Initializing AI model...")
                stages = [
                    (25, "📊 Analyzing clinical measurements..."),
                    (55, "🧬 Evaluating medical history..."),
                    (80, "⚙️ Running prediction model..."),
                    (100, "✅ Finalizing results..."),
                ]
                for pct, msg in stages:
                    time.sleep(0.5)
                    bar.progress(pct, text=msg)
            progress_placeholder.empty()

            result = predict(inputs)
            health_score = compute_health_score(inputs, result["probability"])

            st.session_state.result = result
            st.session_state.inputs = inputs
            st.session_state.health_score = health_score

        if st.session_state.result is not None:
            st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
            render_result(st.session_state.result, st.session_state.inputs, st.session_state.health_score)

    with right:
        render_live_panel(st.session_state.result, st.session_state.health_score)

    st.markdown("<br>", unsafe_allow_html=True)
    render_bottom_section(df, metrics)


if __name__ == "__main__":
    main()