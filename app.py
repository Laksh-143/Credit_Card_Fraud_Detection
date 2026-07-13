"""
app.py
─────────────────────────────────────────────────────────────
Streamlit dashboard for the Fraud Detection Pipeline.

Two tabs:
  1. Model Performance  — metrics, plots, SHAP from saved reports
  2. Live Prediction    — score a single transaction in real time

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Community Cloud
─────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

st.set_page_config(
    page_title = "Fraud Detection AI",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load saved model artifacts. Cached so it only loads once."""
    try:
        from Inference import FraudPredictor
        return FraudPredictor.load("saved_models/")
    except Exception as e:
        return None

predictor = load_predictor()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detection")
    st.markdown("XGBoost + LightGBM Ensemble")
    st.divider()

    model_status = " Model loaded" if predictor else " Model not found"
    st.markdown(f"**Status:** {model_status}")

    if predictor:
        st.markdown(f"**Threshold:** `{predictor.threshold:.3f}`")

    st.divider()
    st.caption("Dataset: Kaggle Synthetic Fraud (1.85M transactions)")
    st.caption("Temporal split — no data leakage")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs([" Model Performance", " Live Prediction"])


# ════════════════════════════════════════════════════════════
# TAB 1 — Model Performance
# ════════════════════════════════════════════════════════════
with tab1:
    st.title("Model Performance Dashboard")
    st.markdown("All metrics evaluated on a **held-out test set** using temporal split — the model never saw these transactions during training.")

    # ── Key metrics ──────────────────────────────────────────
    st.subheader("Key Metrics")

    metrics = {}
    report_path = "reports/evaluation_report.txt"
    if os.path.exists(report_path):
        with open(report_path) as f:
            content = f.read()
        # Parse key numbers from report
        import re
        patterns = {
            "AUC-PR":    r"AUC-PR.*?:\s*([\d.]+)",
            "AUC-ROC":   r"AUC-ROC.*?:\s*([\d.]+)",
            "Recall":    r"Recall.*?:\s*([\d.]+)",
            "Precision": r"Precision.*?:\s*([\d.]+)",
            "F1":        r"F1.*?:\s*([\d.]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("AUC-PR", f"{metrics.get('AUC-PR', '—'):.4f}" if 'AUC-PR' in metrics else "—",
                  help="Primary metric. Measures performance on the fraud class directly.")
    with col2:
        st.metric("AUC-ROC", f"{metrics.get('AUC-ROC', '—'):.4f}" if 'AUC-ROC' in metrics else "—")
    with col3:
        st.metric("Recall", f"{metrics.get('Recall', '—'):.4f}" if 'Recall' in metrics else "—",
                  help="Fraud catch rate — what % of actual fraud did we catch?")
    with col4:
        st.metric("Precision", f"{metrics.get('Precision', '—'):.4f}" if 'Precision' in metrics else "—",
                  help="When we flag fraud, how often are we right?")
    with col5:
        st.metric("F1 Score", f"{metrics.get('F1', '—'):.4f}" if 'F1' in metrics else "—")

    st.divider()

    # ── Algorithm comparison ──────────────────────────────────
    st.subheader("Algorithm Comparison")
    st.markdown("Five algorithms benchmarked on identical data. XGBoost + LightGBM selected as ensemble.")

    comparison_path = "reports/model_comparison.csv"
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        # Highlight best row
        best_idx = comp_df["AUC-PR"].idxmax()
        st.dataframe(
            comp_df.style.highlight_max(subset=["AUC-PR"], color="#d4edda")
                         .format({"AUC-PR": "{:.4f}", "AUC-ROC": "{:.4f}",
                                  "F1": "{:.4f}", "Precision": "{:.4f}",
                                  "Recall": "{:.4f}", "Train Time (s)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart of AUC-PR
        st.bar_chart(
            comp_df.set_index("Model")["AUC-PR"].sort_values(),
            horizontal=True,
        )
    else:
        st.info("Run main.py first to generate model comparison results.")

    st.divider()

    # ── Plots ────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm_path = "reports/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_column_width=True)
        else:
            st.info("Run main.py to generate plots.")

        st.subheader("Score Distribution")
        sd_path = "reports/score_distribution.png"
        if os.path.exists(sd_path):
            st.image(Image.open(sd_path), use_column_width=True)

    with col_right:
        st.subheader("Precision-Recall Curve")
        pr_path = "reports/precision_recall_curve.png"
        if os.path.exists(pr_path):
            st.image(Image.open(pr_path), use_column_width=True)
        else:
            st.info("Run main.py to generate plots.")

        st.subheader("Threshold Cost Curve")
        tc_path = "reports/threshold_cost_curve.png"
        if os.path.exists(tc_path):
            st.image(Image.open(tc_path), use_column_width=True)

    st.divider()

    # ── SHAP ─────────────────────────────────────────────────
    st.subheader("SHAP Feature Importance")
    st.markdown("Which features drive the model's fraud predictions most.")

    shap_path = "reports/shap_xgboost.png"
    if os.path.exists(shap_path):
        st.image(Image.open(shap_path), use_column_width=True)
    else:
        st.info("Run main.py to generate SHAP plots.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Live Prediction
# ════════════════════════════════════════════════════════════
with tab2:
    st.title("Live Transaction Scoring")
    st.markdown("Enter transaction details below to get a real-time fraud probability from the ensemble model.")

    if not predictor:
        st.error("Model not loaded. Make sure `saved_models/` exists and contains trained model files. Run `main.py` first.")
        st.stop()

    # ── Sample transactions ───────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader(" Test Scenarios")
        if st.button(" Normal Transaction", use_container_width=True):
            st.session_state["test_amt"]    = 45.50
            st.session_state["test_cat"]    = "grocery_pos"
            st.session_state["txn_1h"]      = 0
            st.session_state["amt_1h"]      = 0.0
            st.session_state["txn_6h"]      = 1
            st.session_state["amt_6h"]      = 32.0
            st.session_state["txn_24h"]     = 3
            st.session_state["amt_24h"]     = 124.0

        if st.button(" Suspicious Transaction", use_container_width=True):
            st.session_state["test_amt"]    = 987.00
            st.session_state["test_cat"]    = "shopping_net"
            st.session_state["txn_1h"]      = 5
            st.session_state["amt_1h"]      = 2340.0
            st.session_state["txn_6h"]      = 8
            st.session_state["amt_6h"]      = 3100.0
            st.session_state["txn_24h"]     = 10
            st.session_state["amt_24h"]     = 4200.0

    # ── Input form ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        amt        = st.number_input("Transaction Amount ($)",
                                     min_value=0.01, max_value=50000.0,
                                     value=float(st.session_state.get("test_amt", 125.00)))
        category   = st.selectbox("Merchant Category", [
            "shopping_net", "misc_net", "grocery_pos", "shopping_pos",
            "food_dining", "health_fitness", "entertainment",
            "gas_transport", "home", "kids_pets", "personal_care",
            "travel", "misc_pos", "grocery_net"
        ], index=2)
        gender     = st.selectbox("Gender", ["F", "M"])
        state      = st.selectbox("State", [
            "TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA",
            "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "TN"
        ])

    with col2:
        st.subheader("Context Features")
        city_pop   = st.number_input("City Population",
                                     min_value=100, max_value=10_000_000,
                                     value=50000)
        trans_time = st.selectbox("Transaction Hour",
                                  [str(h).zfill(2) for h in range(24)],
                                  index=14)
        job        = st.text_input("Customer Job", value="Software Engineer")
        city       = st.text_input("City", value="Austin")

    st.subheader("Card Velocity (past activity)")
    st.caption("In production these would be auto-fetched from a database. Enter manually for demo.")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        txn_1h  = st.number_input("Transactions in past 1h",  min_value=0, max_value=50,  value=0)
        amt_1h  = st.number_input("Spend in past 1h ($)",     min_value=0.0, max_value=50000.0, value=0.0)
    with vc2:
        txn_6h  = st.number_input("Transactions in past 6h",  min_value=0, max_value=100, value=1)
        amt_6h  = st.number_input("Spend in past 6h ($)",     min_value=0.0, max_value=50000.0, value=85.0)
    with vc3:
        txn_24h = st.number_input("Transactions in past 24h", min_value=0, max_value=200, value=3)
        amt_24h = st.number_input("Spend in past 24h ($)",    min_value=0.0, max_value=50000.0, value=245.0)

    # ── Score ─────────────────────────────────────────────────
    if st.button(" Score Transaction", type="primary", use_container_width=True):
        from datetime import datetime

        transaction = {
            "trans_date_trans_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num":    "4532756279624064",
            "merchant":  "demo_merchant",
            "category":  category,
            "amt":       amt,
            "gender":    gender,
            "city":      city,
            "state":     state,
            "city_pop":  city_pop,
            "dob":       "1985-06-15",
            "job":       job,
            "lat":       30.2672,
            "long":      -97.7431,
            "merch_lat": 30.2672,
            "merch_long":-97.7431,
            # Velocity — from user inputs
            "txn_count_1h":  txn_1h,   "amt_sum_1h":  amt_1h,
            "txn_count_6h":  txn_6h,   "amt_sum_6h":  amt_6h,
            "txn_count_24h": txn_24h,  "amt_sum_24h": amt_24h,
            "is_new_merchant": 1 if txn_1h == 0 and amt > 500 else 0,
            "is_new_state":    0,
        }

        with st.spinner("Scoring transaction ..."):
            try:
                result = predictor.predict_single(transaction)

                # ── Result display ────────────────────────────
                prob      = result["fraud_probability"]
                is_fraud  = result["is_fraud"]
                risk      = result["risk_level"]

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Fraud Probability", f"{prob*100:.2f}%")
                with col_res2:
                    st.metric("Verdict", " FRAUD" if is_fraud else " LEGITIMATE")
                with col_res3:
                    st.metric("Risk Level", risk)

                # Probability gauge
                color = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
                st.progress(prob, text=f"Fraud probability: {prob*100:.1f}%")

                if is_fraud:
                    st.error(f"Transaction flagged as fraud (threshold: {predictor.threshold:.3f})")
                else:
                    st.success(f"Transaction approved (threshold: {predictor.threshold:.3f})")

                # ── SHAP reasons ──────────────────────────────
                st.subheader("Why did the model score this?")
                st.caption("Top features driving this specific prediction (SHAP values)")

                reasons = result.get("top_reasons", [])
                if reasons:
                    reasons_df = pd.DataFrame(reasons)
                    reasons_df["direction"] = reasons_df["shap_value"].apply(
                        lambda x: "↑ Increases fraud risk" if x > 0 else "↓ Reduces fraud risk"
                    )
                    reasons_df["abs_impact"] = reasons_df["shap_value"].abs()
                    reasons_df = reasons_df.sort_values("abs_impact", ascending=False)

                    for _, row in reasons_df.iterrows():
                        color_badge = "" if row["shap_value"] > 0 else "🟢"
                        st.markdown(
                            f"{color_badge} **{row['feature']}** — "
                            f"{row['direction']} (SHAP: `{row['shap_value']:+.4f}`)"
                        )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.caption("Make sure saved_models/ contains all model artifacts.")

    # ── Educational footer ────────────────────────────────────
    st.divider()
    with st.expander("📚 How to read these results"):
        st.markdown("""
**Fraud Probability** — The ensemble model's confidence that this transaction is fraudulent (0–100%).

**Threshold** — The cutoff point optimised using business cost (FN × $250 + FP × $15). Transactions above this are flagged.

**Risk Level**
-  Low (< 30%): Approve automatically
-  Medium (30–50%): Monitor
-  High (50–75%): Flag for review
-  Critical (> 75%): Block and alert

**SHAP values** — Explain WHY the model scored this transaction. A positive SHAP value means that feature pushed the score toward fraud. A negative value means it pushed toward legitimate.

**Velocity features** — The number and total value of transactions on this card in the past 1h/6h/24h. High velocity is the strongest fraud signal.
        """)