# Fraud-Detection-Using-Machine-Learning
This is my academic project and it is best demonstration of Machine learning skill. 

# Explainable Fraud Detection System using Machine Learning

This project presents a complete, end-to-end system for detecting fraudulent financial transactions using a machine learning model that is both highly accurate and fully explainable. The system is built using a Python-based stack and features a back-end API that serves the model and a user-friendly web interface for interaction and demonstration. An alternative, simplified deployment method using a single Streamlit application is also provided.

The core of the project is an XGBoost classifier trained on the PaySim dataset from Kaggle. A key focus of this work was to build a system under realistic data constraints, meaning it does not rely on private recipient information. Its high performance is achieved through advanced feature engineering and a focus on model interpretability using SHAP.

## Key Features

-   **High-Recall Model:** The final XGBoost model achieves **99% recall** on the fraud class, demonstrating its effectiveness at the primary business goal: catching fraudsters.
-   **Advanced Feature Engineering:** The model's intelligence is driven by custom-built features like `senderBalanceError` and `isOrigAccountEmpty` that capture the behavioral signatures of fraud.
-   **Explainable AI (XAI):** The system is not a "black box." By integrating SHAP, we can prove *why* the model makes its decisions, making it transparent and trustworthy.
-   **Interactive Web UI:** A user-friendly front-end built with Streamlit allows for easy manual transaction checks and live demonstrations.
-   **Comprehensive Simulation:** The UI includes a multi-scenario simulation that demonstrates the system's ability to detect various fraud patterns, including velocity attacks and account takeovers.
-   **API-Based Architecture (Primary):** The machine learning model is served via a robust Flask API, separating the model logic from the user interface and allowing for easy integration.
-   **Simplified Streamlit Deployment (Alternative):** An option to run the entire application as a single Streamlit script, embedding model loading and prediction directly within the UI.

## System Architecture

The application offers two deployment options:

### 1. Flask API + Streamlit UI (Original Architecture)

This operates on a client-server architecture:

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

### 2. Streamlit-Only Deployment (Simplified Architecture)

In this setup, the Streamlit application directly loads the machine learning models and performs predictions, eliminating the need for a separate Flask API.

```
+-------------------------------------------------+
|             Streamlit UI (app_ui.py)            |
|  (Loads ML Models directly, performs prediction)|
+-------------------------------------------------+
      |                                 |
      V                                 V
+----------------------+      +----------------------+
|  ML Models (joblib)  |      |  ML Models (joblib)  |
| (XGBoost, IsoForest) |      | (XGBoost, IsoForest) |
+----------------------+      +----------------------+
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

-   **Back-End:** Python, Flask (for original architecture)
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
│   └── app_ui.py               # The Streamlit UI application (connects to Flask API)
│
├── streamlit_deployment/
│   ├── app_ui.py               # The Streamlit UI application (standalone, loads models directly)
│   ├── final_fraud_model.joblib  # Copy of the trained XGBoost model
│   ├── isolation_forest_model.joblib # Copy of the trained Isolation Forest model
│   └── requirements.txt        # Dependencies for Streamlit-only deployment
│
├── notebook/
│   └── Fraud_Detection_Analysis.ipynb  # Your analysis notebook (optional)
│
└── README.md                   # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/agp-369/Fraud-Detection-Using-Machine-Learning.git
    cd Fraud-Detection-Using-Machine-Learning

    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The required packages depend on which deployment method you choose.

    ### For Flask API + Streamlit UI Deployment:
    Create a `requirements.txt` file in your project root (or use the one in `fraud_api` and `fraud_ui` if they exist) with the following content and run `pip install -r requirements.txt`.

    **`requirements.txt` (for Flask API + Streamlit UI):**
    ```
    flask
    pandas
    scikit-learn
    xgboost
    streamlit
    requests
    ```

    ### For Streamlit-Only Deployment:
    Navigate to the `streamlit_deployment` directory and install its specific requirements.

    **`streamlit_deployment/requirements.txt`:**
    ```
    pandas
    scikit-learn
    xgboost
    streamlit
    ```
    Then run:
    ```bash
    cd streamlit_deployment
    pip install -r requirements.txt
    cd .. # Go back to project root if needed
    ```

## How to Run the Application

You have two options for running the application:

### Option 1: Flask API + Streamlit UI (Original Architecture)

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

### Option 2: Streamlit-Only Deployment (Simplified Architecture)

This method runs the entire application from a single Streamlit script.

1.  **Navigate to the `streamlit_deployment` directory:**
    ```bash
    cd streamlit_deployment
    ```

2.  **Ensure model files are present:**
    Make sure `isolation_forest_model.joblib` and `final_fraud_model.joblib` are copied into the `streamlit_deployment` directory.

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app_ui.py
    ```
    This will automatically open a new tab in your web browser with the user interface, usually at `http://localhost:8501`.
## Live Demo   
Experience the Fraud Detection System live: [**Launch Streamlit App**]https://frauddetection-system.streamlit.app/
## Demonstration

You can now use the web interface to check transactions manually or run the comprehensive simulation to see the hybrid detection system in action.

![Screenshot of the Application UI]
<img width="1863" height="919" alt="Screenshot1" src="https://github.com/user-attachments/assets/48f664fc-dde8-4040-89dd-e09c253ed980" />
<img width="1800" height="850" alt="Screenshot2" src="https://github.com/user-attachments/assets/739f748b-2289-4df1-aad6-4dfb16f0e7c4" />

