"""
Live Loan Default Prediction Demo
----------------------------------
Run with:  streamlit run app.py

Requires these 6 files in the same folder (produced by export_model_cells.py
at the end of your notebook):
  loan_model.pkl, scaler.pkl, model_columns.pkl, column_medians.pkl,
  organization_freq_map.pkl, occupation_freq_map.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="\U0001F3E6", layout="centered")

# ---------- Load model + preprocessing artifacts ----------
REQUIRED_FILES = [
    "loan_model.pkl", "scaler.pkl", "model_columns.pkl",
    "column_medians.pkl", "organization_freq_map.pkl", "occupation_freq_map.pkl",
]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing) +
        "\n\nRun the export cell at the end of your notebook first, then place "
        "the generated .pkl files in this same folder."
    )
    st.stop()

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
column_medians = joblib.load("column_medians.pkl")
org_freq_map = joblib.load("organization_freq_map.pkl")
occ_freq_map = joblib.load("occupation_freq_map.pkl")

st.title("Loan Default Risk Predictor")
st.caption("Enter applicant details to get a live prediction from the trained Logistic Regression model.")

# ---------- Input form ----------
with st.form("applicant_form"):
    st.subheader("Financial Details")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income ($)", min_value=1000, value=150000, step=1000)
        credit = st.number_input("Loan Credit Amount ($)", min_value=1000, value=500000, step=1000)
    with col2:
        annuity = st.number_input("Annuity / Monthly Payment ($)", min_value=100, value=25000, step=500)
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

    st.subheader("Applicant Profile")
    col3, col4 = st.columns(2)
    with col3:
        age_years = st.slider("Age (years)", 18, 75, 35)
        years_employed = st.slider("Years Employed", 0, 45, 5)
        gender = st.selectbox("Gender", ["M", "F"])
        own_car = st.selectbox("Owns a Car?", ["Y", "N"])
        own_realty = st.selectbox("Owns Real Estate?", ["Y", "N"])
    with col4:
        education = st.selectbox("Education", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])
        family_status = st.selectbox("Family Status", [
            "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
        ])
        income_type = st.selectbox("Income Type", [
            "Working", "Commercial associate", "Pensioner", "State servant", "Student"
        ])
        housing_type = st.selectbox("Housing Type", [
            "House / apartment", "With parents", "Municipal apartment",
            "Rented apartment", "Office apartment", "Co-op apartment"
        ])

    st.subheader("Credit Bureau Risk Scores")
    st.caption("External credit scores (0 = highest risk, 1 = lowest risk). Leave at default if unknown.")
    col5, col6, col7 = st.columns(3)
    with col5:
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
    with col6:
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
    with col7:
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

    with st.expander("More details (optional)"):
        organization_type = st.selectbox("Organization Type", sorted(org_freq_map.index.tolist()))
        occupation_type = st.selectbox("Occupation Type", sorted(occ_freq_map.index.tolist()))
        weekday = st.selectbox("Application Weekday", [
            "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
        ])
        emergency_state = st.selectbox("Emergency State (building)", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ---------- Build feature row + predict ----------
if submitted:
    # Start every column at its training-set median (safe default for anything
    # not explicitly asked about in the form)
    row = column_medians.reindex(model_columns).fillna(0).astype(float)

    # Core numeric / engineered features
    row["AMT_CREDIT"] = credit
    row["AMT_ANNUITY"] = annuity
    if "AMT_INCOME_TOTAL_log" in row.index:
        row["AMT_INCOME_TOTAL_log"] = np.log1p(income)
    row["CREDIT_INCOME_RATIO"] = credit / income
    row["ANNUITY_INCOME_RATIO"] = annuity / income
    row["DAYS_BIRTH"] = age_years
    row["DAYS_EMPLOYED"] = years_employed
    row["EXT_SOURCE_1"] = ext1
    row["EXT_SOURCE_2"] = ext2
    row["EXT_SOURCE_3"] = ext3

    # Frequency-encoded categoricals
    row["ORGANIZATION_TYPE"] = org_freq_map.get(organization_type, org_freq_map.mean())
    row["OCCUPATION_TYPE"] = occ_freq_map.get(occupation_type, occ_freq_map.mean())

    # One-hot encoded categoricals: zero out the group, then flip the selected one on
    def set_one_hot(prefix, value):
        group_cols = [c for c in model_columns if c.startswith(prefix)]
        for c in group_cols:
            row[c] = 0
        target_col = f"{prefix}{value}"
        if target_col in row.index:
            row[target_col] = 1

    set_one_hot("NAME_CONTRACT_TYPE_", contract_type)
    set_one_hot("CODE_GENDER_", gender)
    set_one_hot("FLAG_OWN_CAR_", own_car)
    set_one_hot("FLAG_OWN_REALTY_", own_realty)
    set_one_hot("NAME_INCOME_TYPE_", income_type)
    set_one_hot("NAME_EDUCATION_TYPE_", education)
    set_one_hot("NAME_FAMILY_STATUS_", family_status)
    set_one_hot("NAME_HOUSING_TYPE_", housing_type)
    set_one_hot("WEEKDAY_APPR_PROCESS_START_", weekday)
    set_one_hot("EMERGENCYSTATE_MODE_", emergency_state)

    # Ensure exact column order the scaler/model expect
    X_new = row.reindex(model_columns).fillna(0).values.reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Default Risk — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Default Risk — estimated probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "This score reflects the trained Logistic Regression model's estimated "
        "probability that this applicant would default on the loan."
    )
