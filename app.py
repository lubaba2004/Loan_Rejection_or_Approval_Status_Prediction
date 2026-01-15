import streamlit as st
import joblib
import numpy as np

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

model = joblib.load(MODEL_PATH)

# App title
st.title("🏦 Loan Approval Prediction App")

st.write("Enter applicant details to predict loan approval status")

# -------- USER INPUTS --------

no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

income_annum = st.number_input("Annual Income", min_value=0, value=500000)

loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)

loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=10)

cibil_score = st.slider("CIBIL Score", 300, 900, 650)

residential_assets_value = st.number_input("Residential Asset Value", min_value=0, value=0)
commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0, value=0)
luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0, value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -------- ENCODING --------

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------- PREDICTION --------

if st.button("Predict Loan Status"):

    input_data = np.array([[ 
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        education,
        self_employed
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {1-probability:.2f})")

