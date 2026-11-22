"""
Credit Card Fraud Detection System - Version 1.0 (initial deployment)
Single-transaction prediction only. Model trained in WM-02AI (tuned XGBoost, threshold from model_config.pkl).
"""
import os
import datetime
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE, "prediction_log.csv")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
    config = joblib.load(os.path.join(BASE, "model_config.pkl"))
    return model, scaler, config


model, scaler, config = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.caption("Version 1.0 - enter the details of one transaction to check whether it looks fraudulent.")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount", min_value=0.0, max_value=100000.0, value=150.0, step=1.0)
    hour = st.number_input("Hour of day (0-23)", min_value=0, max_value=23, value=12, step=1)
    trust = st.number_input("Device trust score (0-100)", min_value=0, max_value=100, value=60, step=1)
    velocity = st.number_input("Transactions in last 24h", min_value=0, max_value=50, value=2, step=1)
with col2:
    age = st.number_input("Cardholder age", min_value=18, max_value=100, value=40, step=1)
    foreign = st.selectbox("Foreign transaction?", ["No", "Yes"])
    mismatch = st.selectbox("Location mismatch?", ["No", "Yes"])

if st.button("Predict", type="primary"):
    raw = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": hour,
        "foreign_transaction": 1 if foreign == "Yes" else 0,
        "location_mismatch": 1 if mismatch == "Yes" else 0,
        "device_trust_score": trust,
        "velocity_last_24h": velocity,
        "cardholder_age": age,
    }])
    X = raw.copy()
    X[["amount", "transaction_hour"]] = scaler.transform(X[["amount", "transaction_hour"]])
    X = X[config["features"]]
    prob = float(model.predict_proba(X)[0, 1])
    is_fraud = prob >= config["threshold"]

    if is_fraud:
        st.error(f"⚠️ FRAUD DETECTED - fraud probability {prob:.1%}")
    else:
        st.success(f"✅ Legitimate - fraud probability {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    row = raw.copy()
    row.insert(0, "timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    row["fraud_probability"] = round(prob, 4)
    row["decision"] = "Fraud" if is_fraud else "Legitimate"
    row["app_version"] = "1.0"
    row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
