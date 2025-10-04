import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import json

# --- Custom transformer used in pipeline ---
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, lower=0.01, upper=0.99):
        self.cols, self.lower, self.upper = cols, lower, upper
        self.bounds_ = {}
    def fit(self, X, y=None):
        X = X.copy()
        for c in self.cols:
            if c in X and pd.api.types.is_numeric_dtype(X[c]):
                lo, hi = X[c].quantile(self.lower), X[c].quantile(self.upper)
                self.bounds_[c] = (lo, hi)
        return self
    def transform(self, X):
        X = X.copy()
        for c,(lo,hi) in self.bounds_.items():
            if c in X and pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].clip(lo, hi)
        return X

# --- Load pipeline + training column names ---
pipe = joblib.load("gbm_pipeline.pkl")
with open("gbm_columns.json", "r") as f:
    train_columns = json.load(f)

st.set_page_config(page_title="Attrition Risk (GBM)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Employee Attrition Risk Prediction")

# --- Inputs ---
col1,col2 = st.columns(2)
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
    edu = st.selectbox("Education Level",["High School","Bachelor’s Degree","Master’s Degree","PhD"])
    marital = st.selectbox("Marital Status",["Married","Single"])
    job_level = st.selectbox("Job Level",["Entry","Mid","Senior"])
    company_size = st.selectbox("Company Size",["Small","Medium"])
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
    # Start with numeric + ordinal features
    row = {
        "Age": age, "Years at Company": years_company, "Monthly Income": monthly_income,
        "Number of Promotions": num_prom, "Distance from Home": dist_home,
        "Company Tenure": tenure, "Number of Dependents": dependents,
        "Work-Life Balance": wlb, "Job Satisfaction": job_sat,
        "Performance Rating": perf, "Company Reputation": reputation,
        "Employee Recognition": recognition
    }

    # Add dummy-encoded features (1 if chosen, else 0)
    dummies = {
        f"Gender_Male": 1 if gender=="Male" else 0,
        f"Job Role_{job_role}": 1,
        f"Education Level_{edu}": 1,
        f"Marital Status_{marital}": 1,
        f"Job Level_{job_level}": 1,
        f"Company Size_{company_size}": 1,
        f"Remote Work_Yes": 1 if remote=="Yes" else 0,
        f"Leadership Opportunities_Yes": 1 if leader=="Yes" else 0,
        f"Innovation Opportunities_Yes": 1 if innov=="Yes" else 0,
        f"Overtime_Yes": 1 if overtime=="Yes" else 0
    }

    row.update(dummies)
    row = pd.DataFrame([row])

    # Align with training column order (fill missing with 0)
    row = row.reindex(columns=train_columns, fill_value=0)

    # --- Predict ---
    prob = float(pipe.predict_proba(row)[:,1][0])
    pred = int(prob >= threshold)

    st.subheader(f"Probability of leaving: **{prob:.2%}**")
    st.write("Prediction:", "🔴 High risk" if pred==1 else "🟢 Low risk")
