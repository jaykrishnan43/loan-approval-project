# Bank Loan Approval Prediction using Explainable AI

This project implements an end-to-end **Bank Loan Approval Prediction System** using machine learning and explainable AI techniques.  
The system predicts whether a loan application should be **Approved or Rejected** and explains each individual decision using a **SHAP waterfall plot**.

The project is designed to reflect **real-world banking workflows**, combining policy rules, predictive modeling, and transparent explanations.

---

## Key Features

- Policy-based screening using **CIBIL score cutoff**
- Loan approval prediction using **XGBoost**
- Feature engineering (loan-income ratio, EMI-income ratio, asset aggregation)
- **SHAP waterfall explanation** for each individual prediction
- Interactive **Streamlit dashboard**
- Clean, modular project structure

---

## Decision Workflow

1. **Policy Rule Enforcement**
   - If CIBIL score is below a defined threshold (default: 650), the loan is automatically rejected.
   - The machine learning model is bypassed.

2. **Machine Learning Prediction**
   - If policy conditions are satisfied, an XGBoost model predicts approval probability.
   - The loan is classified as Approved or Rejected.

3. **Explainability (SHAP Waterfall)**
   - For each prediction, a SHAP waterfall plot explains how each feature contributed to the decision.
   - This ensures transparency, interpretability, and auditability.

---

## Technology Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP, LIME
- Matplotlib
- Streamlit
- Joblib

---

## Project Structure

loan-approval-project/
├── app/
│ └── streamlit_app.py # Streamlit dashboard
├── notebooks/
│ ├── 01_data_preparation.ipynb
│ ├── 02_model_training.ipynb
│ └── 03_explainability.ipynb
├── data/
│ └── dataset.csv # Loan dataset
├── models/
│ ├── xgb_model.pkl # Trained XGBoost model
│ └── feature_columns.json # Feature alignment
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/jaykrishnan43/loan-approval-project.git
cd loan-approval-project

### 2. Install dependencies and run the application
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py


---

```markdown
## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/jaykrishnan43/loan-approval-project.git
cd loan-approval-project

pip install -r requirements.txt
streamlit run app/streamlit_app.py



