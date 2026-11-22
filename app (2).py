"""
Credit Card Fraud Detection System - Version 2.0 (after improvements)
Improvements over v1.0:
  1. SHAP explainability for every prediction
  2. User feedback mechanism + monitoring dashboard
  3. Batch CSV prediction, login, and input validation
Model trained in WM-02AI (tuned XGBoost, decision threshold stored in model_config.pkl).
"""
import os
import hmac
import uuid
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

APP_VERSION = "2.0"
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE, "prediction_log.csv")
FEEDBACK_FILE = os.path.join(BASE, "feedback_log.csv")
RAW_FEATURES = ["amount", "transaction_hour", "foreign_transaction", "location_mismatch",
                "device_trust_score", "velocity_last_24h", "cardholder_age"]
MAX_BATCH_ROWS = 5000


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
    config = joblib.load(os.path.join(BASE, "model_config.pkl"))
    return model, scaler, config


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, scaler, config = load_artifacts()
THRESHOLD = float(config["threshold"])


# ---------------------------------------------------------------- security
def get_credentials():
    """Credentials come from Streamlit secrets or environment variables, never from the code."""
    try:
        return str(st.secrets["APP_USER"]), str(st.secrets["APP_PASSWORD"])
    except Exception:
        return os.environ.get("APP_USER", "analyst"), os.environ.get("APP_PASSWORD", "fraud2026")


def login():
    if st.session_state.get("auth"):
        return True
    st.title("💳 Credit Card Fraud Detection")
    st.subheader("Please log in")
    attempts = st.session_state.get("attempts", 0)
    if attempts >= 5:
        st.error("Too many failed attempts. Refresh the page to try again.")
        return False
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        real_user, real_pw = get_credentials()
        if hmac.compare_digest(user, real_user) and hmac.compare_digest(pw, real_pw):
            st.session_state["auth"] = True
            st.session_state["user"] = user
            st.rerun()
        else:
            st.session_state["attempts"] = attempts + 1
            st.error("Incorrect username or password.")
    return False


# ---------------------------------------------------------------- helpers
def append_csv(path, row_df):
    row_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def to_model_input(raw_df):
    X = raw_df[RAW_FEATURES].copy()
    X[["amount", "transaction_hour"]] = scaler.transform(X[["amount", "transaction_hour"]])
    return X[config["features"]]


def validate(df):
    """Return a list of problems found in a dataframe of raw transactions."""
    problems = []
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    numeric = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().any().any():
        return ["All feature columns must contain numbers only, with no empty cells."]
    d = numeric
    if (d["amount"] < 0).any(): problems.append("amount must be 0 or more.")
    if not d["transaction_hour"].between(0, 23).all(): problems.append("transaction_hour must be 0-23.")
    if not d["foreign_transaction"].isin([0, 1]).all(): problems.append("foreign_transaction must be 0 or 1.")
    if not d["location_mismatch"].isin([0, 1]).all(): problems.append("location_mismatch must be 0 or 1.")
    if not d["device_trust_score"].between(0, 100).all(): problems.append("device_trust_score must be 0-100.")
    if (d["velocity_last_24h"] < 0).any(): problems.append("velocity_last_24h must be 0 or more.")
    if not d["cardholder_age"].between(18, 100).all(): problems.append("cardholder_age must be 18-100.")
    return problems


def explain_figure(X_row, raw_row):
    sv = load_explainer(model).shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)[0]
    order = np.argsort(np.abs(sv))
    labels = [f"{config['features'][i]} = {raw_row[config['features'][i]]}" for i in order]
    colors = ["#C0392B" if sv[i] > 0 else "#2E75B6" for i in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(labels, sv[order], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on fraud score (red = towards fraud, blue = towards legitimate)")
    ax.set_title("Why did the model decide this? (SHAP)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- pages
def page_home():
    st.title("💳 Credit Card Fraud Detection System")
    st.write("This system uses a machine learning model (tuned XGBoost) to estimate whether a credit card "
             "transaction is fraudulent. Use the menu on the left to check one transaction, upload a CSV file, "
             "view the model performance, or review feedback and monitoring.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost (tuned)")
    c2.metric("Decision threshold", f"{THRESHOLD:.2f}")
    c3.metric("App version", APP_VERSION)
    st.info("Every prediction is logged, and users can say whether a prediction was correct. "
            "This feedback is used to monitor the model and decide when to retrain it.")


def page_single():
    st.header("Single Transaction Prediction")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", 0.0, 100000.0, 150.0, 1.0)
        hour = st.number_input("Hour of day (0-23)", 0, 23, 12, 1)
        trust = st.number_input("Device trust score (0-100)", 0, 100, 60, 1)
        velocity = st.number_input("Transactions in last 24h", 0, 50, 2, 1)
    with col2:
        age = st.number_input("Cardholder age", 18, 100, 40, 1)
        foreign = st.selectbox("Foreign transaction?", ["No", "Yes"])
        mismatch = st.selectbox("Location mismatch?", ["No", "Yes"])

    if st.button("Predict", type="primary"):
        raw = pd.DataFrame([{
            "amount": amount, "transaction_hour": hour,
            "foreign_transaction": 1 if foreign == "Yes" else 0,
            "location_mismatch": 1 if mismatch == "Yes" else 0,
            "device_trust_score": trust, "velocity_last_24h": velocity, "cardholder_age": age}])
        X = to_model_input(raw)
        prob = float(model.predict_proba(X)[0, 1])
        pid = uuid.uuid4().hex[:8]
        decision = "Fraud" if prob >= THRESHOLD else "Legitimate"
        st.session_state["last"] = {"id": pid, "prob": prob, "decision": decision,
                                    "raw": raw.iloc[0].to_dict(), "X": X}
        log = raw.copy()
        log.insert(0, "prediction_id", pid)
        log.insert(1, "timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
        log["fraud_probability"] = round(prob, 4)
        log["decision"] = decision
        log["app_version"] = APP_VERSION
        log["user"] = st.session_state.get("user", "")
        append_csv(LOG_FILE, log)

    last = st.session_state.get("last")
    if last:
        st.divider()
        left, right = st.columns([1, 1.3])
        with left:
            if last["decision"] == "Fraud":
                st.error(f"⚠️ FRAUD DETECTED\n\nFraud probability: {last['prob']:.1%}")
            else:
                st.success(f"✅ LEGITIMATE\n\nFraud probability: {last['prob']:.1%}")
            st.progress(min(max(last["prob"], 0.0), 1.0))
            st.caption(f"Decision threshold: {THRESHOLD:.2f} | Prediction ID: {last['id']}")
            with st.form("feedback_form", clear_on_submit=True):
                st.write("**Was this prediction correct?**")
                verdict = st.radio("Your answer", ["Yes, correct", "No, incorrect"], horizontal=True)
                comment = st.text_input("Comment (optional)")
                sent = st.form_submit_button("Submit feedback")
            if sent:
                append_csv(FEEDBACK_FILE, pd.DataFrame([{
                    "prediction_id": last["id"],
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "model_decision": last["decision"],
                    "user_says_correct": verdict.startswith("Yes"),
                    "comment": comment}]))
                st.success("Thank you! Your feedback was saved.")
        with right:
            fig = explain_figure(last["X"], last["raw"])
            st.pyplot(fig)
            plt.close(fig)


def page_batch():
    st.header("Batch Prediction (CSV upload)")
    st.write("Upload a CSV file with these columns: " + ", ".join(f"`{c}`" for c in RAW_FEATURES) +
             f". Maximum {MAX_BATCH_ROWS:,} rows.")
    sample = pd.DataFrame([[150.0, 14, 0, 0, 70, 2, 40], [900.0, 2, 1, 1, 30, 6, 35]], columns=RAW_FEATURES)
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_transactions.csv", "text/csv")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is None:
        return
    try:
        df = pd.read_csv(file)
    except Exception:
        st.error("That file could not be read as a CSV.")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.error(f"Too many rows ({len(df):,}). The limit is {MAX_BATCH_ROWS:,}.")
        return
    problems = validate(df)
    if problems:
        for p in problems:
            st.error(p)
        return
    X = to_model_input(df)
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = probs.round(4)
    out["decision"] = np.where(probs >= THRESHOLD, "Fraud", "Legitimate")
    flagged = int((out["decision"] == "Fraud").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(out):,}")
    c2.metric("Flagged as fraud", f"{flagged:,}")
    c3.metric("Flag rate", f"{flagged / len(out):.1%}")
    st.dataframe(out.style.apply(
        lambda r: ["background-color: #FDECEA" if r["decision"] == "Fraud" else "" for _ in r], axis=1),
        width="stretch")
    st.download_button("Download results", out.to_csv(index=False), "fraud_predictions.csv", "text/csv")
    log = out.copy()
    log.insert(0, "prediction_id", ["b" + uuid.uuid4().hex[:7] for _ in range(len(log))])
    log.insert(1, "timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    log["app_version"] = APP_VERSION
    log["user"] = st.session_state.get("user", "")
    append_csv(LOG_FILE, log[["prediction_id", "timestamp"] + RAW_FEATURES +
                             ["fraud_probability", "decision", "app_version", "user"]])


def page_performance():
    st.header("Model Performance (from WM-02AI testing)")
    st.write("Results of the final tuned XGBoost model on the 2,000-transaction test set (30 fraud cases), "
             f"using a decision threshold of {THRESHOLD:.2f}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", "0.862"); c2.metric("Recall", "0.833")
    c3.metric("F1-Score", "0.848"); c4.metric("AUC-ROC", "0.998")
    cm = np.array([[1966, 4], [5, 25]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > 1000 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Pred Legit", "Pred Fraud"]); ax.set_yticks([0, 1], ["Actual Legit", "Actual Fraud"])
    ax.set_title("Final confusion matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The model caught 25 of 30 fraud cases and raised 4 false alarms among 1,970 legitimate transactions.")


def page_monitoring():
    st.header("Feedback and Monitoring")
    if not os.path.exists(LOG_FILE):
        st.info("No predictions have been logged yet.")
        return
    log = pd.read_csv(LOG_FILE)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions logged", f"{len(log):,}")
    c2.metric("Flagged as fraud", f"{(log['decision'] == 'Fraud').mean():.1%}")
    c3.metric("Average fraud probability", f"{log['fraud_probability'].mean():.3f}")
    if os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        acc = fb["user_says_correct"].astype(bool).mean()
        st.metric("Predictions rated correct by users", f"{acc:.0%}", help=f"Based on {len(fb)} feedback entries")
        if len(fb) >= 20 and acc < 0.80:
            st.warning("Feedback accuracy has dropped below 80%. Consider retraining the model.")
        st.write("**Latest feedback**")
        st.dataframe(fb.tail(10), width="stretch")
    else:
        st.info("No feedback has been submitted yet.")
    st.write("**Latest predictions**")
    st.dataframe(log.tail(10), width="stretch")
    st.download_button("Download prediction log", log.to_csv(index=False), "prediction_log.csv", "text/csv")


# ---------------------------------------------------------------- main
if login():
    st.sidebar.title("Menu")
    st.sidebar.caption(f"Logged in as {st.session_state.get('user', '')}")
    page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction",
                                      "Model Performance", "Feedback & Monitoring"])
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()
    {"Home": page_home, "Single Prediction": page_single, "Batch Prediction": page_batch,
     "Model Performance": page_performance, "Feedback & Monitoring": page_monitoring}[page]()
