import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
from streamlit_lottie import st_lottie
import requests
import plotly.express as px

# Load the credit risk model
model = joblib.load("model/credit_risk_model.pkl")

# Load Lottie animation from URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

low_risk_lottie = load_lottie_url("https://lottie.host/8fd3322b-dc86-4653-8591-d54f56a39062/GUrA6L5WfB.json")
high_risk_lottie = load_lottie_url("https://lottie.host/88f6f2b3-bb9d-4c43-b82d-31c9ad1df3b4/D5Dd0LpjQf.json")

st.set_page_config(page_title="AI Risk & Cloud Advisor", layout="wide")

# Tab layout for multiple apps
tab1, tab2 = st.tabs(["💳 Credit Risk Predictor", "☁️ Cloud Cost Optimizer"])

# Credit Risk Predictor Tab
with tab1:
    st.title("💳 Credit Risk Predictor")

    # Sidebar for user input
    with st.sidebar:
        st.header("🔍 Manual Input")
        age = st.slider("Age", 18, 70, 30)
        income = st.number_input("Annual Income ($)", min_value=1000, value=50000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amount = st.number_input("Loan Amount ($)", min_value=500, value=15000)
        loan_duration = st.slider("Loan Duration (Months)", 6, 60, 24)
        employment_status = st.selectbox("Employment Status", ["Unemployed", "Part-time", "Full-time"])

        employment_map = {"Unemployed": 0, "Part-time": 1, "Full-time": 2}
        emp_status_val = employment_map[employment_status]

        input_data = np.array([[age, income, credit_score, loan_amount, loan_duration, emp_status_val]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of default

    # Display prediction and chart
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Input Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Age", "Income", "Credit Score", "Loan Amount", "Loan Duration", "Employment"],
            "Value": [age, income, credit_score, loan_amount, loan_duration, employment_status]
        })
        st.dataframe(summary_df, use_container_width=True)

        fig = px.bar(summary_df, x="Feature", y="Value", title="User Input Overview", text="Value", color="Feature")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🧠 Prediction Result")
        if prediction == 0:
            st.success(f"Low Risk - {round((1 - probability) * 100)}% confidence")
            st_lottie(low_risk_lottie, height=250, key="low")
        else:
            st.error(f"Default Risk - {round(probability * 100)}% confidence")
            st_lottie(high_risk_lottie, height=250, key="high")

    st.markdown("---")
    st.subheader("📂 Batch Prediction via CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            batch_input = df[["Age", "Income", "CreditScore", "LoanAmount", "LoanDuration", "EmploymentStatus"]].copy()
            batch_input["EmploymentStatus"] = batch_input["EmploymentStatus"].map(employment_map)
            preds = model.predict(batch_input)
            probs = model.predict_proba(batch_input)[:, 1]
            df["Prediction"] = ["Low Risk" if p == 0 else "Default Risk" for p in preds]
            df["Default Probability"] = probs
            st.success("✅ Batch prediction completed")
            st.dataframe(df)

            st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error in processing file: {e}")

# Cloud Cost Optimizer Tab (Placeholder)
with tab2:
    st.title("☁️ Cloud Cost Advisor (Coming Soon!)")
    st.info("This section will let you upload cloud usage data, analyze cost optimization strategies, and get AI-driven recommendations.")
    st_lottie("https://lottie.host/df4cd615-05e1-481c-bf31-35096470edaf/xplPllJEvQ.json", height=300)