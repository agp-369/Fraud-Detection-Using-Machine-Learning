import joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# 1. Initialize the Flask Application
app = Flask(__name__)

# 2. Load BOTH models
iso_forest_model = joblib.load('isolation_forest_model.joblib')
xgb_model = joblib.load('final_fraud_model.joblib')

# 3. Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])

    # --- Feature Engineering (This must match the training process) ---
    df['senderBalanceError'] = df['oldbalanceOrg'] + df['amount'] - df['newbalanceOrig']
    df['isOrigAccountEmpty'] = (df['newbalanceOrig'] == 0).astype(int)
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7
    if 'amount_deviation_from_avg' not in df.columns: df['amount_deviation_from_avg'] = 0
    if 'is_new_recipient' not in df.columns: df['is_new_recipient'] = 1
    if 'time_since_last_transaction' not in df.columns: df['time_since_last_transaction'] = 0
    if 'transactions_in_last_hour' not in df.columns: df['transactions_in_last_hour'] = 1
    if 'type_CASH_OUT' not in df.columns: df['type_CASH_OUT'] = 0
    if 'type_TRANSFER' not in df.columns: df['type_TRANSFER'] = 0

    # --- STAGE 1: Unsupervised Anomaly Detection (Fast Filter) ---
    # CORRECTED: Use the same features the XGBoost model was trained on for consistency
    # The Isolation Forest is robust to extra features.
    all_features = [
        'amount', 'isOrigAccountEmpty', 'senderBalanceError', 'hour_of_day',
        'day_of_week', 'amount_deviation_from_avg', 'is_new_recipient',
        'time_since_last_transaction', 'type_CASH_OUT', 'type_TRANSFER'
    ]
    
    # Ensure all columns are present
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    iso_forest_pred = iso_forest_model.predict(df[all_features])

    if iso_forest_pred[0] == 1:
        return jsonify({'isFraud': 0, 'stage': 1, 'details': 'Passed initial anomaly scan.'})

    # --- STAGE 2: Supervised Classification (Expert Analyst) ---
    final_df = df[all_features]
    xgb_prediction = xgb_model.predict(final_df)

    return jsonify({'isFraud': int(xgb_prediction[0]), 'stage': 2, 'details': 'Flagged as anomaly and confirmed by deep analysis.'})

# 4. Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)