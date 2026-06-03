# ============================================================
# dashboard.py — User Dashboard for BTC Prediction API
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="BTC Signal Dashboard",
    page_icon="₿",
    layout="wide",
)

API_URL = "http://localhost:8001"

st.markdown("""
<style>
.stApp { background: #07090F; color: #E8ECF4; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.card {
    background: linear-gradient(145deg, #0D1120 0%, #111827 100%);
    border: 1px solid #1E2540;
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1rem;
}
.title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F0F4FF;
}
.subtitle {
    color: #7A84A0;
    margin-bottom: 2rem;
}
.label {
    color: #7A84A0;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.value {
    font-size: 3rem;
    font-weight: 800;
    font-family: monospace;
}
.metric {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: monospace;
}
.neutral { color: #F0B429; }
.up { color: #00D4AA; }
.down { color: #FF4B6E; }
.info {
    color: #8A94B0;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


def fetch_latest_prediction():
    url = f"{API_URL}/predict/latest"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def signal_style(prediction: str):
    if prediction == "UP":
        return "up", "▲ UP", "Model expects upward movement"
    if prediction == "DOWN":
        return "down", "▼ DOWN", "Model expects downward movement"
    return "neutral", "— NEUTRAL", "Model expects no strong directional move"


st.markdown("<div class='title'>₿ BTC Direction Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>LightGBM Soft-Regime model with HMM regime probabilities</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### API Settings")
    st.write(f"Backend: `{API_URL}`")
    st.markdown("---")
    st.markdown("### Endpoints")
    st.write("GET `/health`")
    st.write("GET `/predict/latest`")
    st.markdown("---")
    st.caption("Not financial advice. Academic ML project.")

if st.button("↻ Refresh Prediction"):
    st.cache_data.clear()

try:
    result = fetch_latest_prediction()
except Exception as e:
    st.error(f"Cannot connect to FastAPI backend: {e}")
    st.stop()

prediction = result.get("prediction", "UNKNOWN")
probability = result.get("probability", 0)
model_type = result.get("model_type", "Unknown")
date = result.get("date", "Unknown")

css_class, signal_text, explanation = signal_style(prediction)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="label">Latest Signal</div>
        <div class="value {css_class}">{signal_text}</div>
        <br>
        <div class="info">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="label">Confidence</div>
        <div class="metric">{probability * 100:.2f}%</div>
        <br>
        <div class="label">Data Date</div>
        <div class="metric">{date}</div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="label">Prediction Raw</div>
        <div class="metric">{result.get("prediction_raw", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="label">Model</div>
        <div class="metric" style="font-size:1rem;">{model_type}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="label">Backend Status</div>
        <div class="metric up">ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <div class="label">Interpretation</div>
    <div class="info">
        <b>UP</b> means the model expects Bitcoin to move above the positive threshold.<br>
        <b>DOWN</b> means the model expects Bitcoin to move below the negative threshold.<br>
        <b>NEUTRAL</b> means the model does not expect a strong enough move in either direction.
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Raw API Response"):
    st.json(result)