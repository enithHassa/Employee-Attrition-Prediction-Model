import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

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

# --- Load pipeline ---
pipe = joblib.load("gbm_pipeline.pkl")  # ✅ path simplified for GitHub deployment

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
    row = pd.DataFrame([{
        "Age": age, "Years at Company": years_company, "Monthly Income": monthly_income,
        "Number of Promotions": num_prom, "Distance from Home": dist_home,
        "Company Tenure": tenure, "Number of Dependents": dependents,
        "Gender": gender, "Job Role": job_role, "Education Level": edu,
        "Marital Status": marital, "Job Level": job_level, "Company Size": company_size,
        "Remote Work": remote, "Leadership Opportunities": leader, "Innovation Opportunities": innov,
        "Overtime": overtime, "Work-Life Balance": wlb, "Job Satisfaction": job_sat,
        "Performance Rating": perf, "Company Reputation": reputation, "Employee Recognition": recognition
    }])

    # Force numeric only on numerical columns
    numeric_features = [
        "Age","Years at Company","Monthly Income","Number of Promotions",
        "Distance from Home","Company Tenure","Number of Dependents"
    ]
    for col in numeric_features:
        row[col] = pd.to_numeric(row[col], errors="coerce")

   # Ensure ordinal categories match training
ordinal_map = {
    "Work-Life Balance": ["Poor","Below Average","Good","Excellent"],
    "Job Satisfaction": ["Very Low","Low","Medium","High"],
    "Performance Rating": ["Low","Below Average","Average","High"],
    "Company Reputation": ["Very Poor","Poor","Good","Excellent"],
    "Employee Recognition": ["Very Low","Low","Medium","High"]
}

for col, cats in ordinal_map.items():
    if col in row:
        row[col] = pd.Categorical(row[col], categories=cats, ordered=True)

    prob = float(pipe.predict_proba(row)[:,1][0])
    pred = int(prob >= threshold)

    st.subheader(f"Probability of leaving: **{prob:.2%}**")
    st.write("Prediction:", "🔴 High risk" if pred==1 else "🟢 Low risk")
