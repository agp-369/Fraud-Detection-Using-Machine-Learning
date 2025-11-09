import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import shap

# -----------------------------------------------------------
# ✅ Streamlit Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Fraud Detection System",
    page_icon="🏆",
    layout="wide"
)

# -----------------------------------------------------------
# Load Models and Create Explainer
# -----------------------------------------------------------
@st.cache_resource
def load_resources():
    import os
    MODEL_DIR = os.path.dirname(__file__)
    iso_forest_model = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest_model.joblib'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'final_fraud_model.joblib'))
    
    # Create the SHAP explainer from the loaded model
    shap_explainer = shap.TreeExplainer(xgb_model)
    
    return iso_forest_model, xgb_model, shap_explainer

iso_forest_model, xgb_model, shap_explainer = load_resources()

# -----------------------------------------------------------
# Core Prediction Function
# -----------------------------------------------------------
def predict_transaction(data):
    df = pd.DataFrame(data, index=[0])

    # --- Feature Engineering ---
    df['senderBalanceError'] = df['oldbalanceOrg'] + df['amount'] - df['newbalanceOrig']
    df['isOrigAccountEmpty'] = (df['newbalanceOrig'] == 0).astype(int)
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7

    for col in ['amount_deviation_from_avg', 'is_new_recipient', 'time_since_last_transaction',
                'transactions_in_last_hour', 'type_CASH_OUT', 'type_TRANSFER']:
        if col not in df.columns:
            df[col] = 0

    all_features = [
        'amount', 'isOrigAccountEmpty', 'senderBalanceError', 'hour_of_day',
        'day_of_week', 'amount_deviation_from_avg', 'is_new_recipient',
        'time_since_last_transaction', 'type_CASH_OUT', 'type_TRANSFER'
    ]
    
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    # --- Stage 1: Isolation Forest ---
    iso_forest_pred = iso_forest_model.predict(df[all_features])
    if iso_forest_pred[0] == 1:
        return {'isFraud': 0, 'stage': 1}

    # --- Stage 2: XGBoost ---
    xgb_prediction = xgb_model.predict(df[all_features])

    # --- SHAP Explanation for Stage 2 ---
    shap_values = shap_explainer.shap_values(df[all_features])
    shap_explanation = dict(zip(df[all_features].columns, shap_values[0]))

    return {
        'isFraud': int(xgb_prediction[0]),
        'stage': 2,
        'shap_explanation': shap_explanation
    }

# -----------------------------------------------------------
# UI and the rest of the app logic...
# (The rest of the file remains the same as the last correct version)
# -----------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = {}

st.title("Hybrid Two-Stage Fraud Detection System 🏆")
# ... (The rest of the UI code is identical to the last fully-featured version)