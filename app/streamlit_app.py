# app/streamlit_app.py
# Run from project root:
# streamlit run app/streamlit_app.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# ======================
# Absolute paths
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")


# ======================
# Page config
# ======================
st.set_page_config(page_title="Loan Approval Explanation", layout="centered")
st.title("Loan Approval Prediction with SHAP Waterfall Explanation")


# ======================
# Load model and columns
# ======================
@st.cache_resource
def load_model_and_columns():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        cols = json.load(f)
    return model, cols


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, FEATURE_COLUMNS = load_model_and_columns()
explainer = load_explainer(model)


# ======================
# Helper functions
# ======================
def build_input_row(
    income, loan_amount, loan_term, cibil,
    res_assets, com_assets, lux_assets, bank_assets,
    dependents, education, self_emp
):
    eps = 1e-9

    loan_income_ratio = loan_amount / (income + eps)
    emi_proxy = loan_amount / (loan_term + eps)
    emi_income_ratio = emi_proxy / (income + eps)
    asset_total = res_assets + com_assets + lux_assets + bank_assets

    return pd.DataFrame([{
        "no_of_dependents": dependents,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": res_assets,
        "commercial_assets_value": com_assets,
        "luxury_assets_value": lux_assets,
        "bank_asset_value": bank_assets,
        "loan_income_ratio": loan_income_ratio,
        "emi_proxy": emi_proxy,
        "emi_income_ratio": emi_income_ratio,
        "asset_total": asset_total,
        "education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "self_employed_Yes": 1 if self_emp == "Yes" else 0,
    }])


def align_features(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


def compute_shap(explainer, X):
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).ravel()[0])

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    return base_val, np.array(shap_vals).reshape(-1)


# ======================
# Sidebar inputs
# ======================
st.sidebar.header("Applicant Inputs")

income = st.sidebar.number_input("Annual Income", min_value=0, value=5_000_000, step=100_000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=15_000_000, step=100_000)
loan_term = st.sidebar.number_input("Loan Term (years)", min_value=1, value=10)
cibil = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900, value=700)

res_assets = st.sidebar.number_input("Residential Assets Value", min_value=0, value=5_000_000, step=100_000)
com_assets = st.sidebar.number_input("Commercial Assets Value", min_value=0, value=3_000_000, step=100_000)
lux_assets = st.sidebar.number_input("Luxury Assets Value", min_value=0, value=2_000_000, step=100_000)
bank_assets = st.sidebar.number_input("Bank Asset Value", min_value=0, value=3_000_000, step=100_000)

dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

policy_cutoff = st.sidebar.slider("Minimum CIBIL Required", 300, 900, 650)


# ======================
# Prediction + SHAP Waterfall
# ======================
if st.button("Predict and Explain"):
    if cibil < policy_cutoff:
        st.error("Loan Rejected")
        st.info(f"Policy rule applied: CIBIL score below {policy_cutoff}.")
    else:
        raw_input = build_input_row(
            income, loan_amount, loan_term, cibil,
            res_assets, com_assets, lux_assets, bank_assets,
            dependents, education, self_emp
        )

        X = align_features(raw_input, FEATURE_COLUMNS)

        prob = model.predict_proba(X)[0, 1]
        decision = "Approved" if prob >= 0.5 else "Rejected"

        if decision == "Approved":
            st.success(f"Loan Approved | Probability: {prob:.3f}")
        else:
            st.error(f"Loan Rejected | Probability: {prob:.3f}")

        base_val, shap_vec = compute_shap(explainer, X)

        shap_exp = shap.Explanation(
            values=shap_vec,
            base_values=base_val,
            data=X.iloc[0].values,
            feature_names=X.columns.tolist()
        )

        st.subheader("SHAP Waterfall Explanation for This Applicant")
        fig = plt.figure()
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig, clear_figure=True)
