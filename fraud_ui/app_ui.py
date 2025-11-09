import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt  # For plotting SHAP values
import numpy as np  # For numerical operations

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid Fraud Detection System",
    page_icon="🏆",
    layout="wide"
)

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = {}

# --- Helper Functions ---
def get_hybrid_prediction(api_data):
    api_url = "http://127.0.0.1:5000/predict"
    try:
        response = requests.post(api_url, json=api_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return None


def update_log_display(placeholder):
    all_history_for_display = []
    for user, history in st.session_state.history.items():
        for tx in history:
            all_history_for_display.append({
                "Timestamp": tx['timestamp'].strftime("%H:%M:%S"),
                "User ID": user,
                "Type": tx['type'],
                "Amount": f"₹{tx['amount']:,.2f}",
                "Prediction": tx['prediction']
            })
    if not all_history_for_display:
        placeholder.info("No transactions in this session yet.")
    else:
        history_df = pd.DataFrame(all_history_for_display).sort_values(
            by="Timestamp", ascending=False
        )
        history_df.reset_index(drop=True, inplace=True)
        history_df.index += 1
        placeholder.dataframe(history_df, use_container_width=True)


# --- Function to display SHAP explanation ---
def display_shap_explanation(shap_explanation_data):
    if not shap_explanation_data:
        st.info("No detailed explanation available for this transaction (passed Stage 1).")
        return

    st.subheader("Why this decision? (SHAP Explanation)")

    # Convert dict to DataFrame for easier plotting
    shap_df = pd.DataFrame(list(shap_explanation_data.items()), columns=['Feature', 'SHAP Value'])
    shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values(by='Absolute SHAP Value', ascending=False).head(5)  # Top 5 features

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['red' if val > 0 else 'blue' for val in shap_df['SHAP Value']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.set_xlabel("SHAP Value (Impact on Model Output)")
    ax.set_title("Top 5 Feature Contributions")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Prevent memory leaks

    st.markdown(
        """
        <small>
        <span style="color:red;">Red bars</span> indicate features pushing towards **FRAUD**.<br>
        <span style="color:blue;">Blue bars</span> indicate features pushing towards **LEGITIMATE**.
        </small>
        """,
        unsafe_allow_html=True
    )


# --- Main App ---
st.title("Hybrid Two-Stage Fraud Detection System 🏆")
st.write(
    "This system uses a two-stage process: a fast anomaly scan followed by a deep analysis for suspicious transactions."
)
st.markdown("---")

col_input, col_history = st.columns([0.5, 0.5], gap="large")

# --- Define Placeholders ---
with col_history:
    st.header("Live Transaction Log")
    log_placeholder = st.empty()

with col_input:
    st.header("Transaction Controls")
    with st.form("transaction_form"):
        user_id = st.text_input("User ID", "USER_12345")
        transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
        amount = st.number_input("Amount (₹)", min_value=0.01, value=1000.0, format="%.2f")
        oldbalanceOrg = st.number_input("Current Balance (₹)", min_value=0.0, value=50000.0, format="%.2f")
        step = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=10)

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            submitted = st.form_submit_button("Check Transaction")
        with form_col2:
            run_simulation = st.form_submit_button("🚀 Run Comprehensive Simulation")

    st.markdown("---")
    st.header("Prediction Result")
    result_placeholder = st.empty()
    shap_explanation_placeholder = st.empty()  # New placeholder for SHAP explanation

# --- Initial Display of Log ---
update_log_display(log_placeholder)


# --- Function to process and display results ---
def process_and_display(api_data, user_id, tx_type, amt, placeholder, is_simulation=False):
    now = datetime.now()
    if user_id not in st.session_state.history:
        st.session_state.history[user_id] = []
    user_history = st.session_state.history[user_id]
    tx_in_last_minute = sum(1 for tx in user_history if now - tx['timestamp'] < timedelta(minutes=1))

    # Clear previous SHAP explanation
    shap_explanation_placeholder.empty()

    with placeholder.container():
        if is_simulation:
            st.info(f"Simulating transaction for {user_id}...")
            st.write("**Frontend Inputs:**")
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("User ID", user_id)
            f_col2.metric("Amount", f"₹{amt:,.2f}")
            f_col3.metric("Hour", api_data['step'])

            st.write("**Backend Engineered Features:**")
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric(
                "Sender Balance Error",
                f"₹{api_data['oldbalanceOrg'] + api_data['amount'] - api_data['newbalanceOrig']:,.2f}"
            )
            b_col2.metric("Account Emptied?", "Yes" if api_data['newbalanceOrig'] == 0 else "No")
            b_col3.metric("Tx in Last Hour", api_data['transactions_in_last_hour'])
            st.markdown("---")

        with st.spinner("Stage 1: Performing rapid anomaly scan..."):
            time.sleep(2)
            prediction_data = get_hybrid_prediction(api_data)

    if prediction_data is not None:
        prediction = prediction_data['isFraud']
        stage = prediction_data['stage']
        shap_explanation = prediction_data.get('shap_explanation', None)

        st.session_state.history[user_id].append({
            "timestamp": now,
            "type": tx_type,
            "amount": amt,
            "prediction": "🚨 FRAUD" if prediction == 1 else "✅ Legitimate"
        })
        update_log_display(log_placeholder)

        with placeholder.container():
            st.subheader("Final Prediction")
            res_col1, res_col2 = st.columns([0.7, 0.3])

            with res_col1:
                if stage == 1:
                    st.success("Stage 1 Result: Passed anomaly scan. Transaction is LEGITIMATE.", icon="✅")
                else:
                    with st.spinner("Stage 2: Anomaly detected! Performing deep analysis..."):
                        time.sleep(2)
                    if prediction == 1:
                        st.error("Stage 2 Result: Deep analysis confirmed transaction is FRAUDULENT.", icon="🚨")
                    else:
                        st.warning("Stage 2 Result: Anomaly found, but deep analysis confirmed it is LEGITIMATE.", icon="⚠️")

            with res_col2:
                if tx_in_last_minute + 1 > 2:
                    st.warning(f"**Velocity Alert:** {tx_in_last_minute + 1} tx/min", icon="⚠️")

        # Display SHAP explanation if available
        if shap_explanation:
            display_shap_explanation(shap_explanation)


# --- Manual Submission Logic ---
if submitted:
    if amount > oldbalanceOrg:
        result_placeholder.error("Error: Transaction amount cannot be greater than the current balance.")
    else:
        newbalanceOrig = oldbalanceOrg - amount
        manual_api_data = {
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "type_CASH_OUT": 1 if transaction_type == "CASH_OUT" else 0,
            "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0,
            "nameOrig": user_id,
            "nameDest": "M_MANUAL",
            "time_since_last_transaction": 0,
            "transactions_in_last_hour": 1
        }
        process_and_display(manual_api_data, user_id, transaction_type, amount, result_placeholder)

# --- Comprehensive Simulation Logic ---
if run_simulation:
    comprehensive_simulation_data = [
        {"user": "USER_A", "type": "CASH_OUT", "amount": 2000.0, "oldbalance": 80000.0, "step": 10},
        {"user": "USER_B", "type": "CASH_OUT", "amount": 15000.0, "oldbalance": 100000.0, "step": 14},
        {"user": "USER_B", "type": "CASH_OUT", "amount": 25000.0, "oldbalance": 85000.0, "step": 14},
        {"user": "USER_B", "type": "CASH_OUT", "amount": 20000.0, "oldbalance": 60000.0, "step": 14},
        {"user": "USER_D", "type": "TRANSFER", "amount": 1500000.0, "oldbalance": 1500000.0, "step": 2},
    ]

    for tx in comprehensive_simulation_data:
        sim_user_id = tx['user']
        newbalance = tx['oldbalance'] - tx['amount'] if tx['oldbalance'] > tx['amount'] else 0
        sim_api_data = {
            "step": tx['step'],
            "amount": tx['amount'],
            "oldbalanceOrg": tx['oldbalance'],
            "newbalanceOrig": newbalance,
            "type_CASH_OUT": 1 if tx['type'] == "CASH_OUT" else 0,
            "type_TRANSFER": 1 if tx['type'] == "TRANSFER" else 0,
            "nameOrig": sim_user_id,
            "nameDest": "M_SIM",
            "time_since_last_transaction": 0,
            "transactions_in_last_hour": 1
        }
        process_and_display(sim_api_data, sim_user_id, tx['type'], tx['amount'], result_placeholder, is_simulation=True)
        time.sleep(6)

    result_placeholder.empty()
    st.success("Simulation complete!")
