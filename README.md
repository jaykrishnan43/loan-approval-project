# Loan Approval Prediction

This project predicts whether a bank loan application should be approved or rejected using machine learning and explains each decision using SHAP waterfall plots.

## Overview
The system follows a realistic banking workflow:
- Apply a policy rule using CIBIL score
- Predict loan approval using an XGBoost model
- Explain the decision using a SHAP waterfall explanation

## Features
- Policy-based screening using minimum CIBIL score
- Loan approval prediction using XGBoost
- SHAP waterfall explanation for each individual prediction
- Interactive Streamlit dashboard

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- LIME
- Streamlit
- Matplotlib

## Project Structure
loan-approval-project/
app/
notebooks/
data/
models/
requirements.txt
README.md
.gitignore

## How to Run
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## Explainability
Each loan decision is explained using a SHAP waterfall plot, showing how individual features contribute to approval or rejection.

## Author
Jayakrishnan K
https://github.com/jaykrishnan43
