# Credit Card Fraud Detection System

AI Software & Developer Level 5 - Work Experience Modules WM-01AI, WM-02AI and WM-03AI.

A machine learning web application that estimates whether a credit card transaction is fraudulent.
The model is a tuned XGBoost classifier trained on 10,000 transactions (151 fraud cases, 1.51%).
On the 2,000-transaction test set it reached precision 0.862, recall 0.833, F1-score 0.848 and AUC-ROC 0.998
(decision threshold 0.45).

## Features (version 2.0)
- Login page (credentials stored in Streamlit secrets, not in the code)
- Single transaction prediction with fraud probability
- SHAP explanation showing why a transaction was flagged
- Batch prediction from a CSV file, with input validation and downloadable results
- Feedback button ("Was this prediction correct?") saved to feedback_log.csv
- Monitoring page: prediction counts, flag rate, feedback accuracy and a retraining warning
- Every prediction is logged to prediction_log.csv

## Project structure
- app.py - the Streamlit application
- fraud_model.pkl, scaler.pkl, model_config.pkl - saved model, scaler and settings from the WM-02AI notebook
- requirements.txt - Python libraries
- README.md - this file

## Run locally
1. Install Python 3.10 or newer.
2. Install the libraries: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Log in with the credentials set in the APP_USER and APP_PASSWORD environment variables
   (defaults for local testing only: analyst / fraud2026).

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click "Create app", choose the repository, set the main file to app.py and deploy.
4. In the app's Settings > Secrets add:
   APP_USER = "your_username"
   APP_PASSWORD = "your_strong_password"

## Version history
- v1.0 - single prediction and logging only
- v2.0 - SHAP explanations, feedback, monitoring page, batch prediction, login and validation

## Limitations
- Streamlit Community Cloud storage is temporary, so log files are lost when the app restarts. A production
  system would store logs in a database.
- The model was trained on a small dataset with only 30 fraud cases in the test set.
- merchant_category is not used by the model.
