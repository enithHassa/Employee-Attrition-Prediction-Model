import joblib
import pandas as pd
import streamlit as st
import json

# --- Load pipeline ---
pipe = joblib.load("gbm_pipeline.pkl")

# --- Load training columns (so Streamlit builds same dummy structure) ---
with open("gbm_columns.json", "r") as f:
    train_columns = json.load(f)

st.set_page_config(page_title="Attrition Risk (GBM)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Employee Attrition Risk Prediction")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age",18,60,30)
    years_company = st.number_input("Years at Company",0,60,3)
    monthly_income = st.number_input("Monthly Income",0,20000,6000,step=100)
    num_prom = st.number_input("Number of Promotions",0,20,0)
    dist_home = st.number_input("Distance from Home (miles)",0,300,10)
    tenure = st.number_input("Company Tenure (years)",0,200,10)
    dependents = st.number_input("Number of Dependents",0,10,0)
with col2:
    gender = st.selectbox("Gender",["Male","Female"])
    job_role = st.selectbox("Job Role",["Finance","Healthcare","Technology","Education","Media"])
    edu = st.selectbox("Education Level",["High School","Associate Degree","Bachelor’s Degree","Master’s Degree","PhD"])
    marital = st.selectbox("Marital Status",["Married","Single","Divorced"])
    job_level = st.selectbox("Job Level",["Entry","Mid","Senior"])
    company_size = st.selectbox("Company Size",["Small","Medium","Large"])
    remote = st.selectbox("Remote Work",["No","Yes"])
    leader = st.selectbox("Leadership Opportunities",["No","Yes"])
    innov = st.selectbox("Innovation Opportunities",["No","Yes"])
    overtime = st.selectbox("Overtime",["No","Yes"])

wlb = st.selectbox("Work-Life Balance",["Poor","Below Average","Good","Excellent"])
job_sat = st.selectbox("Job Satisfaction",["Very Low","Low","Medium","High"])
perf = st.selectbox("Performance Rating",["Low","Below Average","Average","High"])
reputation = st.selectbox("Company Reputation",["Very Poor","Poor","Good","Excellent"])
recognition = st.selectbox("Employee Recognition",["Very Low","Low","Medium","High"])
threshold = st.slider("Decision Threshold",0.05,0.95,0.5,0.01)

# --- Prediction ---
if st.button("Predict risk"):
    # Start with all 0 columns
    row = pd.DataFrame([[0]*len(train_columns)], columns=train_columns)

    # Fill numeric values
    row["Age"] = age
    row["Years at Company"] = years_company
    row["Monthly Income"] = monthly_income
    row["Number of Promotions"] = num_prom
    row["Distance from Home"] = dist_home
    row["Company Tenure"] = tenure
    row["Number of Dependents"] = dependents

    # Fill binary dummies
    if f"Gender_{gender}" in row.columns: row[f"Gender_{gender}"] = 1
    if f"Job Role_{job_role}" in row.columns: row[f"Job Role_{job_role}"] = 1
    if f"Education Level_{edu}" in row.columns: row[f"Education Level_{edu}"] = 1
    if f"Marital Status_{marital}" in row.columns: row[f"Marital Status_{marital}"] = 1
    if f"Job Level_{job_level}" in row.columns: row[f"Job Level_{job_level}"] = 1
    if f"Company Size_{company_size}" in row.columns: row[f"Company Size_{company_size}"] = 1
    if f"Remote Work_{remote}" in row.columns: row[f"Remote Work_{remote}"] = 1
    if f"Leadership Opportunities_{leader}" in row.columns: row[f"Leadership Opportunities_{leader}"] = 1
    if f"Innovation Opportunities_{innov}" in row.columns: row[f"Innovation Opportunities_{innov}"] = 1
    if "Overtime" in row.columns: row["Overtime"] = 1 if overtime=="Yes" else 0

    # Ordinal features (already numeric in splits, so we map them to numbers)
    ordinal_maps = {
        "Work-Life Balance": {"Poor":0,"Below Average":1,"Good":2,"Excellent":3},
        "Job Satisfaction": {"Very Low":0,"Low":1,"Medium":2,"High":3},
        "Performance Rating": {"Low":0,"Below Average":1,"Average":2,"High":3},
        "Company Reputation": {"Very Poor":0,"Poor":1,"Good":2,"Excellent":3},
        "Employee Recognition": {"Very Low":0,"Low":1,"Medium":2,"High":3},
    }
    row["Work-Life Balance"] = ordinal_maps["Work-Life Balance"][wlb]
    row["Job Satisfaction"] = ordinal_maps["Job Satisfaction"][job_sat]
    row["Performance Rating"] = ordinal_maps["Performance Rating"][perf]
    row["Company Reputation"] = ordinal_maps["Company Reputation"][reputation]
    row["Employee Recognition"] = ordinal_maps["Employee Recognition"][recognition]

    # --- Predict ---
    prob = float(pipe.predict_proba(row)[:,1][0])
    pred = int(prob >= threshold)

    st.subheader(f"Probability of leaving: **{prob:.2%}**")
    st.write("Prediction:", "🔴 High risk" if pred==1 else "🟢 Low risk")
