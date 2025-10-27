# Fraud-Detection-Using-Machine-Learning
This is my academic project and it is best demonstration of Machine learning skill. 

# Explainable Fraud Detection System using Machine Learning

This project presents a complete, end-to-end system for detecting fraudulent financial transactions using a machine learning model that is both highly accurate and fully explainable. The system is built using a Python-based stack and features a back-end API that serves the model and a user-friendly web interface for interaction and demonstration.

The core of the project is an XGBoost classifier trained on the PaySim dataset from Kaggle. A key focus of this work was to build a system under realistic data constraints, meaning it does not rely on private recipient information. Its high performance is achieved through advanced feature engineering and a focus on model interpretability using SHAP.

## Key Features

-   **High-Recall Model:** The final XGBoost model achieves **99% recall** on the fraud class, demonstrating its effectiveness at the primary business goal: catching fraudsters.
-   **Advanced Feature Engineering:** The model's intelligence is driven by custom-built features like `senderBalanceError` and `isOrigAccountEmpty` that capture the behavioral signatures of fraud.
-   **Explainable AI (XAI):** The system is not a "black box." By integrating SHAP, we can prove *why* the model makes its decisions, making it transparent and trustworthy.
-   **Interactive Web UI:** A user-friendly front-end built with Streamlit allows for easy manual transaction checks and live demonstrations.
-   **Comprehensive Simulation:** The UI includes a multi-scenario simulation that demonstrates the system's ability to detect various fraud patterns, including velocity attacks and account takeovers.
-   **API-Based Architecture:** The machine learning model is served via a robust Flask API, separating the model logic from the user interface and allowing for easy integration.

## System Architecture

The application operates on a simple but robust client-server architecture:

1.  **Back-End (Flask API):**
    -   Loads the pre-trained Isolation Forest and XGBoost models.
    -   Exposes a `/predict` endpoint that receives transaction data.
    -   Implements the two-stage detection logic: a fast anomaly scan followed by a deep analysis.
    -   Returns the final prediction as a JSON response.

2.  **Front-End (Streamlit UI):**
    -   Provides a clean user interface for entering transaction details.
    -   Sends the user input to the Flask API.
    -   Receives the prediction and displays the result in a clear, user-friendly format, including visual alerts for the detection stage and velocity.

```
+----------------+      +---------------------+      +----------------------+
|  Streamlit UI  | <--> |      Flask API      | <--> |  ML Models (joblib)  |
| (app_ui.py)    |      | (app.py)            |      | (XGBoost, IsoForest) |
+----------------+      +---------------------+      +----------------------+
```

## Methodology Workflow

The project followed a comprehensive machine learning pipeline:

1.  **Data Analysis:** The PaySim dataset was analyzed, revealing that fraud only occurred in `TRANSFER` and `CASH_OUT` transactions. The data was filtered accordingly.
2.  **Feature Engineering:** New, highly predictive features were created from the raw data to capture behavioral patterns without using private recipient information.
3.  **Model Training:** A comparative analysis was performed between a Random Forest baseline and an XGBoost classifier. The models were trained to handle the extreme class imbalance by using class weights (`scale_pos_weight`), prioritizing recall.
4.  **Model Evaluation:** The XGBoost model was selected as the final model due to its superior recall (99%) on the unseen test set.
5.  **Explainability Analysis:** SHAP was used to analyze the final XGBoost model, confirming that our engineered features were the most important drivers of its predictions.

## Results

The final XGBoost model demonstrated excellent performance, prioritizing the critical task of catching fraud.

| Class     | Precision | Recall     | F1-Score |
| :-------- | :-------- | :--------- | :------- |
| **Fraud** | 0.27      | **0.99**   | 0.43     |

The 99% recall proves the model's effectiveness. The lower precision is an accepted and well-understood trade-off in fraud detection, where minimizing missed frauds is the top priority.

## Technologies Used

-   **Back-End:** Python, Flask
-   **Machine Learning:** Pandas, Scikit-learn, XGBoost, SHAP
-   **Front-End:** Streamlit
-   **Data Analysis:** Jupyter Notebook (or Google Colab)

## File Structure

```
.
├── fraud_api/
│   ├── app.py                  # The Flask API server
│   ├── final_fraud_model.joblib  # The trained XGBoost model
│   └── isolation_forest_model.joblib # The trained Isolation Forest model
│
├── fraud_ui/
│   └── app_ui.py               # The Streamlit UI application
│
├── notebook/
│   └── Fraud_Detection_Analysis.ipynb  # Your analysis notebook (optional)
│
└── README.md                   # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.

    **`requirements.txt`:**
    ```
    flask
    pandas
    scikit-learn
    xgboost
    streamlit
    requests
    ```

## How to Run the Application

This system requires two terminals running simultaneously.

1.  **Start the Back-End API:**
    Open a terminal, navigate to the `fraud_api` directory, and run:
    ```bash
    cd fraud_api
    python app.py
    ```
    You should see output indicating the server is running on `http://127.0.0.1:5000`.

2.  **Start the Front-End UI:**
    Open a **second** terminal, navigate to the `fraud_ui` directory, and run:
    ```bash
    cd fraud_ui
    streamlit run app_ui.py
    ```
    This will automatically open a new tab in your web browser with the user interface, usually at `http://localhost:8501`.

## Demonstration

You can now use the web interface to check transactions manually or run the comprehensive simulation to see the hybrid detection system in action.

![Screenshot of the Application UI](screenshot.png)
