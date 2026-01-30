# Bank Loan Approval Prediction

This project predicts whether a bank loan should be **Approved or Rejected** using machine learning and explains each decision using a **SHAP waterfall plot**.

The system follows a realistic banking workflow by applying policy rules before machine learning predictions.

---

## What the Project Does

- Applies a **policy rule** (minimum CIBIL score)
- Predicts loan approval using **XGBoost**
- Explains each prediction using **SHAP waterfall explanation**
- Provides an interactive **Streamlit dashboard**

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP, LIME
- Streamlit
- Matplotlib

---

## Project Structure

loan-approval-project/
├── app/ # Streamlit app
├── notebooks/ # Data prep, training, explainability
├── data/ # Dataset
├── models/ # Trained model and feature columns
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Explainability

Each loan decision is explained using a SHAP waterfall plot, showing how individual features contribute to approval or rejection.

Author

Jayakrishnan K
