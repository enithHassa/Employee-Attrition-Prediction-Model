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

# Little helper so we never forget Overtime
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
    overtime = st.selectbox("Overtime", ["No","Yes"])  # <- raw categorical in splits, but expected as a numeric column

wlb = st.selectbox("Work-Life Balance", ["Poor","Below Average","Good","Excellent"])
job_sat = st.selectbox("Job Satisfaction", ["Very Low","Low","Medium","High"])
perf = st.selectbox("Performance Rating", ["Low","Below Average","Average","High"])
reputation = st.selectbox("Company Reputation", ["Very Poor","Poor","Good","Excellent"])
recognition = st.selectbox("Employee Recognition", ["Very Low","Low","Medium","High"])
threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.5, 0.01)

show_debug = st.checkbox("Show debug info")

# ---------- Prediction ----------
if st.button("Predict risk"):
    # 1) Start with all-zeros row for every training column
    row = pd.DataFrame([[0]*len(train_columns)], columns=train_columns)

    # 2) Numeric / ordinal columns (present as numeric columns in your splits)
    #    These MUST be in train_columns, otherwise your JSON and model are out of sync.
    numeric_ordinal_values = {
        "Age": age,
        "Years at Company": years_company,
        "Monthly Income": monthly_income,
        "Number of Promotions": num_prom,
        "Distance from Home": dist_home,
        "Company Tenure": tenure,
        "Number of Dependents": dependents,
        # Ordinals were stored as numeric in splits (0..3) — we map them now:
        "Work-Life Balance": {"Poor":0,"Below Average":1,"Good":2,"Excellent":3}[wlb],
        "Job Satisfaction": {"Very Low":0,"Low":1,"Medium":2,"High":3}[job_sat],
        "Performance Rating": {"Low":0,"Below Average":1,"Average":2,"High":3}[perf],
        "Company Reputation": {"Very Poor":0,"Poor":1,"Good":2,"Excellent":3}[reputation],
        "Employee Recognition": {"Very Low":0,"Low":1,"Medium":2,"High":3}[recognition],
    }
    for k, v in numeric_ordinal_values.items():
        if k in row.columns:
            row.at[0, k] = v  # set value only if model used that column

    # 3) One-hot dummies (set only the column(s) that exist in training)
    def set_dummy(prefix: str, value: str):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row.at[0, col] = 1

    set_dummy("Gender", "Male" if gender == "Male" else "Female")  # if Female dummy was dropped, it just won't be set
    # Job Role dummies used in your splits (Finance/Healthcare/Media/Technology; Education was the baseline)
    set_dummy("Job Role", job_role)
    # Education Level dummies present in your splits (no Associate Degree there)
    set_dummy("Education Level", edu)
    # Marital Status dummies present in your splits (Married, Single; Divorced was baseline)
    set_dummy("Marital Status", marital)
    # Job Level dummies (Mid, Senior; Entry baseline)
    set_dummy("Job Level", job_level)
    # Company Size dummies (Small, Medium; Large baseline)
    set_dummy("Company Size", company_size)
    # Binary “Yes” flags that were one-hot encoded in splits
    if "Remote Work_Yes" in row.columns:
        row.at[0, "Remote Work_Yes"] = 1 if remote == "Yes" else 0
    if "Leadership Opportunities_Yes" in row.columns:
        row.at[0, "Leadership Opportunities_Yes"] = 1 if leader == "Yes" else 0
    if "Innovation Opportunities_Yes" in row.columns:
        row.at[0, "Innovation Opportunities_Yes"] = 1 if innov == "Yes" else 0

    # 4) Overtime — your splits kept it as a separate column named exactly "Overtime"
    #    And later we trained the model with it **as numeric** (1/0). Ensure it’s present & numeric:
    assert_required_column(train_columns, "Overtime")
    row["Overtime"] = 1 if overtime == "Yes" else 0

    if show_debug:
        st.write("Non-zero columns being sent to model:", list(row.columns[(row != 0).any(axis=0)]))
        st.write("Has 'Overtime' column?", "Overtime" in row.columns)
        st.write("Overtime value:", int(row.at[0, "Overtime"]) if "Overtime" in row else "N/A")

    # 5) Predict
    prob = float(pipe.predict_proba(row)[:, 1][0])
    pred = int(prob >= threshold)

    st.subheader(f"Probability of leaving: **{prob:.2%}**")
    st.write("Prediction:", "🔴 High risk" if pred == 1 else "🟢 Low risk")

        # --- Dynamic tone based on probability ---
    if prob >= 0.8:
        st.warning("⚠️ Very high risk of attrition — immediate HR attention recommended.")
    elif prob >= 0.6:
        st.info("🔶 Moderate to high risk — review key employee satisfaction factors.")
    elif prob <= 0.3:
        st.success("✅ Very low attrition risk — employee appears stable and satisfied.")
    else:
        st.info("🟢 Low to moderate risk — maintain positive engagement and monitor occasionally.")

    # --- Rule-based recommendations ---
    suggestions = []

    # ----- HIGH RISK CASES -----
    if pred == 1:
        # Job satisfaction
        if job_sat in ["Very Low", "Low"]:
            suggestions.append("Improve job satisfaction through recognition, workload management, or career development.")
        elif job_sat == "Medium":
            suggestions.append("Consider gathering feedback to identify satisfaction issues.")

        # Work-life balance
        if wlb in ["Poor", "Below Average"]:
            suggestions.append("Encourage flexible hours or partial remote work to enhance work-life balance.")

        # Compensation fairness
        if monthly_income < 4000 and job_level in ["Mid", "Senior"]:
            suggestions.append("Review compensation fairness relative to experience and job level.")
        elif monthly_income < 2500 and job_level == "Entry":
            suggestions.append("Consider revising entry-level pay to stay competitive.")

        # Performance & development
        if perf in ["Low", "Below Average"]:
            suggestions.append("Provide mentoring or upskilling to improve performance.")
        elif perf == "Average":
            suggestions.append("Encourage further training to boost performance.")

        # Recognition
        if recognition in ["Very Low", "Low"]:
            suggestions.append("Enhance recognition programs — appreciation improves retention.")

        # Company reputation
        if reputation in ["Very Poor", "Poor"]:
            suggestions.append("Work on strengthening company culture and internal communication.")

        # Education vs income mismatch
        if edu in ["Master’s Degree", "PhD"] and monthly_income < 5000:
            suggestions.append("Reevaluate compensation for highly qualified employees.")

        # Tenure and promotion stagnation
        if years_company > 8 and num_prom == 0:
            suggestions.append("Consider promotions or new responsibilities for long-tenured employees.")

        # Distance and remote options
        if dist_home > 30:
            suggestions.append("Explore hybrid work — long commutes often lead to attrition.")
        if remote == "No" and wlb in ["Poor", "Below Average"]:
            suggestions.append("Introduce partial remote options to reduce burnout.")

        # Growth & innovation opportunities
        if leader == "No":
            suggestions.append("Offer leadership opportunities to enhance engagement.")
        if innov == "No":
            suggestions.append("Encourage involvement in innovation projects to increase motivation.")

    # ----- LOW RISK CASES -----
    else:
        if prob < 0.2:
            st.success("🌟 Excellent retention indicators — maintain current work environment.")
        elif prob < 0.4:
            st.info("👍 Stable employee — small improvements could lower risk further.")

        # Reinforce positives
        if job_sat == "High":
            suggestions.append("Continue providing career growth and recognition programs.")
        if wlb in ["Good", "Excellent"]:
            suggestions.append("Maintain balanced workloads and flexible arrangements.")
        if recognition == "High":
            suggestions.append("Keep up recognition culture — proven to boost loyalty.")

        # Continuous improvement
        if perf == "Average":
            suggestions.append("Encourage further skill-building to reach high performance.")
        if years_company > 5 and num_prom < 1:
            suggestions.append("Plan career progression paths to maintain motivation.")

    # --- Display recommendations ---
    if suggestions:
        st.markdown("### 💡 Recommendations:")
        for s in suggestions:
            icon = "⚠️" if pred == 1 else "✅"
            st.markdown(f"{icon} {s}")

