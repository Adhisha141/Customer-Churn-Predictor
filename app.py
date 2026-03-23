"""
app.py  —  Customer Churn Predictor
Run with:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .risk-high  { background:#fee2e2; color:#991b1b; border-radius:8px; padding:12px 20px; font-weight:600; font-size:1.1rem; }
  .risk-low   { background:#d1fae5; color:#065f46; border-radius:8px; padding:12px 20px; font-weight:600; font-size:1.1rem; }
  .metric-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; }
  .token-high { background:#fecaca; border-radius:4px; padding:2px 6px; margin:2px; display:inline-block; font-size:0.95rem; }
  .token-med  { background:#fde68a; border-radius:4px; padding:2px 6px; margin:2px; display:inline-block; font-size:0.95rem; }
  .token-low  { background:#e0e7ff; border-radius:4px; padding:2px 6px; margin:2px; display:inline-block; font-size:0.95rem; }
  .section-header { font-size:1rem; font-weight:600; color:#1e293b; margin-bottom:8px; margin-top:16px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.caption("Hybrid ML + NLP model: predict churn from customer chat logs and profile data")
st.divider()

# ── Sidebar: Tabular inputs ─────────────────────────────────────────────────────
st.sidebar.header("Customer Profile (ML model)")
st.sidebar.caption("These features go into the XGBoost classifier.")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.number_input("Total charges ($)", value=float(tenure * monthly_charges))
contract = st.sidebar.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
senior = st.sidebar.checkbox("Senior citizen")

ml_weight = st.sidebar.slider(
    "ML weight in fusion", 0.0, 1.0, 0.4,
    help="How much weight to give the tabular model vs the text model."
)

# ── Main: Text input ────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.3, 1])

with col_l:
    st.markdown('<div class="section-header">💬 Customer chat / support log</div>', unsafe_allow_html=True)
    default_text = (
        "I've been a customer for 3 months but the service keeps dropping. "
        "I called support twice and got no help. I'm really thinking about cancelling "
        "and switching to a competitor. This is really frustrating."
    )
    chat_text = st.text_area("Paste customer message or chat log:", value=default_text, height=160)

    run_btn = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running models…"):
        from utils.predictor import predict_text, predict_tabular, fuse_predictions

        # NLP prediction
        nlp_result = predict_text(chat_text)
        text_prob = nlp_result["churn_probability"]

        # Tabular feature encoding (mirrors training preprocessing)
        contract_map  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map  = {"DSL": 0, "Fiber optic": 1, "No": 2}
        payment_map   = {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }
        tab_features = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract_map[contract],
            "InternetService": internet_map[internet],
            "PaymentMethod": payment_map[payment],
            "SeniorCitizen": int(senior),
        }
        ml_result = predict_tabular(tab_features)
        tab_prob = ml_result.get("churn_probability")

        fused = fuse_predictions(text_prob, tab_prob, text_weight=1 - ml_weight)

    # ── Results ─────────────────────────────────────────────────────────────
    with col_l:
        st.divider()
        risk_class = "risk-high" if fused >= 0.5 else "risk-low"
        risk_emoji = "🔴" if fused >= 0.5 else "🟢"
        risk_label = "HIGH churn risk" if fused >= 0.5 else "LOW churn risk"
        st.markdown(
            f'<div class="{risk_class}">{risk_emoji} {risk_label} — {fused*100:.1f}% probability</div>',
            unsafe_allow_html=True,
        )

        # Score breakdown
        st.markdown('<div class="section-header">Score breakdown</div>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("NLP (text) score", f"{text_prob*100:.1f}%", delta=None)
        mc2.metric(
            "ML (profile) score",
            f"{tab_prob*100:.1f}%" if tab_prob is not None else "N/A",
        )
        mc3.metric("Fused score", f"{fused*100:.1f}%")
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

        acc1, acc2 = st.columns(2)
        acc1.metric("ML Model Accuracy", "86%")
        acc2.metric("NLP Model Accuracy", "91%")

        # Token highlights
        st.markdown('<div class="section-header">🔤 Key words driving the text prediction</div>', unsafe_allow_html=True)
        tokens = nlp_result["token_scores"]
        if tokens:
            max_score = max(t["score"] for t in tokens) or 1.0
            html_tokens = ""
            for t in tokens:
                norm = t["score"] / max_score
                if norm >= 0.6:
                    cls = "token-high"
                elif norm >= 0.3:
                    cls = "token-med"
                else:
                    cls = "token-low"
                html_tokens += f'<span class="{cls}">{t["token"]}</span> '
            st.markdown(html_tokens, unsafe_allow_html=True)
            st.caption("🔴 High attention  🟡 Medium  🔵 Low — based on DistilBERT [CLS] attention weights")

    with col_r:
        st.divider()
        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fused * 100,
            title={"text": "Churn Risk %", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#ef4444" if fused >= 0.5 else "#22c55e"},
                "steps": [
                    {"range": [0, 40],  "color": "#d1fae5"},
                    {"range": [40, 65], "color": "#fef9c3"},
                    {"range": [65, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            }
        ))
        gauge.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)

        # Attention bar chart
        if tokens:
            sorted_tokens = sorted(tokens, key=lambda x: x["score"], reverse=True)[:10]
            bar = go.Figure(go.Bar(
                x=[t["score"] for t in sorted_tokens],
                y=[t["token"] for t in sorted_tokens],
                orientation="h",
                marker_color="#f87171",
            ))
            bar.update_layout(
                title="Top attention tokens",
                height=300,
                margin=dict(t=40, b=10, l=10, r=10),
                yaxis={"autorange": "reversed"},
                xaxis_title="Attention weight",
            )
            st.plotly_chart(bar, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "This app uses:\n"
    "1) A text model to read customer messages\n"
    "2) A data model to analyze customer details\n"
    "Both results are combined to predict churn risk.\n"
    "You can retrain the models using train_nlp.py and train_ml.py.\n"
)
