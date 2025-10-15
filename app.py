import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import json

# ---------- Custom transformer (needed to unpickle the pipeline) ----------
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
        for c, (lo, hi) in self.bounds_.items():
            if c in X and pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].clip(lo, hi)
        return X

# ---------- Load pipeline + training column order ----------
pipe = joblib.load("gbm_pipeline.pkl")
with open("gbm_columns.json", "r") as f:
    train_columns = json.load(f)

st.set_page_config(page_title="Attrition Risk (GBM)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Employee Attrition Risk Prediction")

# Little helper for consistency check
def assert_required_column(cols_list, col_name: str):
    if col_name not in cols_list:
        st.error(
            f"Model expects a column '{col_name}' but it's missing from gbm_columns.json.\n"
            f"→ Rebuild and re-save pipeline + columns JSON in the same run."
        )
        st.stop()

# ---------- UI ----------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 60, 30)
    years_company = st.number_input("Years at Company", 0, 60, 3)
    monthly_income = st.number_input("Monthly Income", 0, 20000, 6000, step=100)
    num_prom = st.number_input("Number of Promotions", 0, 20, 0)
    dist_home = st.number_input("Distance from Home (miles)", 0, 300, 10)
    tenure = st.number_input("Company Tenure (years)", 0, 200, 10)
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", ["Finance","Healthcare","Technology","Education","Media"])
    edu = st.selectbox("Education Level", ["High School","Bachelor’s Degree","Master’s Degree","PhD"])
    marital = st.selectbox("Marital Status", ["Married","Single"])
    job_level = st.selectbox("Job Level", ["Entry","Mid","Senior"])
    company_size = st.selectbox("Company Size", ["Small","Medium"])
    remote = st.selectbox("Remote Work", ["No","Yes"])
    leader = st.selectbox("Leadership Opportunities", ["No","Yes"])
    innov = st.selectbox("Innovation Opportunities", ["No","Yes"])
    overtime = st.selectbox("Overtime", ["No","Yes"])

wlb = st.selectbox("Work-Life Balance", ["Poor","Below Average","Good","Excellent"])
job_sat = st.selectbox("Job Satisfaction", ["Very Low","Low","Medium","High"])
perf = st.selectbox("Performance Rating", ["Low","Below Average","Average","High"])
reputation = st.selectbox("Company Reputation", ["Very Poor","Poor","Good","Excellent"])
recognition = st.selectbox("Employee Recognition", ["Very Low","Low","Medium","High"])
threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.5, 0.01)

show_debug = st.checkbox("Show debug info")

# ---------- Prediction ----------
if st.button("Predict risk"):
    # 1) Start with zero row for every training column
    row = pd.DataFrame([[0]*len(train_columns)], columns=train_columns)

    # 2) Numeric & Ordinal columns
    numeric_ordinal_values = {
        "Age": age,
        "Years at Company": years_company,
        "Monthly Income": monthly_income,
        "Number of Promotions": num_prom,
        "Distance from Home": dist_home,
        "Company Tenure": tenure,
        "Number of Dependents": dependents,
        "Work-Life Balance": {"Poor":0,"Below Average":1,"Good":2,"Excellent":3}[wlb],
        "Job Satisfaction": {"Very Low":0,"Low":1,"Medium":2,"High":3}[job_sat],
        "Performance Rating": {"Low":0,"Below Average":1,"Average":2,"High":3}[perf],
        "Company Reputation": {"Very Poor":0,"Poor":1,"Good":2,"Excellent":3}[reputation],
        "Employee Recognition": {"Very Low":0,"Low":1,"Medium":2,"High":3}[recognition],
    }
    for k, v in numeric_ordinal_values.items():
        if k in row.columns:
            row.at[0, k] = v

    # 3) One-hot dummies
    def set_dummy(prefix: str, value: str):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row.at[0, col] = 1

    set_dummy("Gender", gender)
    set_dummy("Job Role", job_role)
    set_dummy("Education Level", edu)
    set_dummy("Marital Status", marital)
    set_dummy("Job Level", job_level)
    set_dummy("Company Size", company_size)
    if "Remote Work_Yes" in row.columns:
        row.at[0, "Remote Work_Yes"] = 1 if remote == "Yes" else 0
    if "Leadership Opportunities_Yes" in row.columns:
        row.at[0, "Leadership Opportunities_Yes"] = 1 if leader == "Yes" else 0
    if "Innovation Opportunities_Yes" in row.columns:
        row.at[0, "Innovation Opportunities_Yes"] = 1 if innov == "Yes" else 0

    assert_required_column(train_columns, "Overtime")
    row["Overtime"] = 1 if overtime == "Yes" else 0

    if show_debug:
        st.write("Non-zero columns sent to model:", list(row.columns[(row != 0).any(axis=0)]))
        st.write("Overtime value:", row["Overtime"].iloc[0])

    # 4) Predict
    prob = float(pipe.predict_proba(row)[:, 1][0])
    pred = int(prob >= threshold)

    st.subheader(f"Probability of leaving: **{prob:.2%}**")
    st.write("Prediction:", "🔴 High risk" if pred == 1 else "🟢 Low risk")

    # --- Dynamic tone based on probability ---
    if prob >= 0.8:
        st.warning("⚠️ Very high attrition risk detected — urgent HR intervention recommended.")
    elif prob >= 0.6:
        st.info("🔶 Moderate-to-high risk — proactive retention strategies advised.")
    elif prob <= 0.3:
        st.success("✅ Very low attrition risk — employee is likely satisfied and stable.")
    else:
        st.info("🟢 Low-to-moderate risk — continue engagement and monitoring.")

    # --- Intelligent, rule-based recommendations ---
    suggestions = []

    # ----- HIGH RISK CASES -----
    if pred == 1:
        st.warning("🚨 **High attrition risk detected** — recommended HR review.")

        # Job Satisfaction
        if job_sat in ["Very Low", "Low"]:
            suggestions.append("Improve job satisfaction via recognition, career development, or workload balance.")
        elif job_sat == "Medium":
            suggestions.append("Conduct one-on-one sessions to identify job satisfaction pain points.")

        # Work-Life Balance
        if wlb in ["Poor", "Below Average"]:
            suggestions.append("Encourage flexible working hours or hybrid options to improve work-life balance.")

        # Compensation Fairness
        if monthly_income < 4000 and job_level in ["Mid", "Senior"]:
            suggestions.append("Reassess salary fairness compared to industry averages for experienced staff.")
        elif monthly_income < 2500 and job_level == "Entry":
            suggestions.append("Review entry-level pay rates to ensure competitiveness and motivation.")

        # Performance and Growth
        if perf in ["Low", "Below Average"]:
            suggestions.append("Offer mentoring, performance improvement plans, or training programs.")
        elif perf == "Average":
            suggestions.append("Encourage training or certifications to boost performance and confidence.")

        # Recognition & Engagement
        if recognition in ["Very Low", "Low"]:
            suggestions.append("Enhance recognition initiatives — even small gestures improve engagement.")

        # Company Reputation
        if reputation in ["Very Poor", "Poor"]:
            suggestions.append("Improve internal communication and transparency to rebuild trust in company image.")

        # Education vs Income Mismatch
        if edu in ["Master’s Degree", "PhD"] and monthly_income < 5000:
            suggestions.append("Highly educated employees may feel undervalued — review compensation alignment.")

        # Tenure and Promotions
        if years_company > 8 and num_prom == 0:
            suggestions.append("Consider new challenges, project leadership, or promotion opportunities.")
        elif years_company > 5 and num_prom < 1:
            suggestions.append("Discuss growth prospects to prevent stagnation feelings.")

        # Distance from Work
        if dist_home > 30:
            suggestions.append("Long commutes increase stress — explore remote or hybrid work arrangements.")

        # Leadership & Innovation
        if leader == "No":
            suggestions.append("Provide leadership chances to build ownership and motivation.")
        if innov == "No":
            suggestions.append("Encourage involvement in innovation or creative initiatives to build engagement.")

        # Remote Work Impact
        if remote == "No" and wlb in ["Poor", "Below Average"]:
            suggestions.append("Introduce partial remote-work options to reduce burnout risk.")

        # Overtime
        if overtime == "Yes":
            suggestions.append("Reduce excessive overtime — fatigue contributes to attrition.")

    # ----- LOW RISK CASES -----
    else:
        st.success("✅ Low attrition risk detected — employee appears engaged and content.")

        if prob < 0.2:
            st.balloons()
            suggestions.append("Maintain the current positive work environment — retention outlook excellent.")
        elif prob < 0.4:
            suggestions.append("Stable engagement — continue recognition and feedback initiatives.")

        # Reinforce Positives
        if job_sat == "High":
            suggestions.append("Continue growth opportunities and feedback — this employee is thriving.")
        if wlb in ["Good", "Excellent"]:
            suggestions.append("Keep flexible and balanced workload to sustain motivation.")
        if recognition == "High":
            suggestions.append("Maintain recognition programs — employees value acknowledgment.")

        # Encourage improvement even in low-risk
        if perf == "Average":
            suggestions.append("Encourage professional development to push performance from average to high.")
        if years_company > 5 and num_prom < 1:
            suggestions.append("Plan mid-term career progression or mentoring to sustain engagement.")
        if innov == "No":
            suggestions.append("Offer small innovation or cross-team projects to retain curiosity.")

    # --- Display all suggestions ---
    if suggestions:
        st.markdown("### 💡 Recommendations:")
        for s in suggestions:
            icon = "⚠️" if pred == 1 else "✅"
            st.markdown(f"{icon} {s}")
