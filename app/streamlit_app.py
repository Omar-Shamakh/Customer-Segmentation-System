"""
Interactive Streamlit app for real-time customer segment prediction.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "kmeans_pipeline.joblib")
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cluster_profiles.csv")

CLUSTER_PERSONAS = {
    0: "Older, budget-conscious households with children — low engagement",
    1: "Younger, lower-income customers — lowest spenders, low engagement",
    2: "Mature, comfortable-income customers — strong spenders",
    3: "High-income, top spenders with the highest campaign responsiveness — VIP segment",
    4: "High-income, high-spending customers with few/no children — low campaign engagement",
}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_profiles():
    return pd.read_csv(PROFILES_PATH, index_col="Cluster")


def main():
    st.set_page_config(page_title="Customer Segmentation", layout="centered")
    st.title("🛍️ Customer Segmentation — Real-Time Prediction")
    st.write("Enter a customer's profile to predict their segment using the trained K-Means model.")

    pipeline = load_pipeline()
    profiles = load_profiles()

    st.header("Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 40)
        income = st.number_input("Annual Income ($)", 0, 200000, 50000, step=1000)
        education = st.selectbox("Education", ["Undergraduate", "Master", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Partnered"])
        total_children = st.slider("Number of Children", 0, 3, 0)
        family_size = st.slider("Family Size", 1, 6, 1 + total_children)

    with col2:
        total_spending = st.number_input("Total Spending ($, last 2 years)", 0, 3000, 500)
        total_purchases = st.slider("Total Purchases", 0, 40, 10)
        recency = st.slider("Days Since Last Purchase", 0, 100, 30)
        tenure_days = st.slider("Customer Tenure (days)", 0, 1000, 300)
        campaigns_accepted = st.slider("Campaigns Accepted (past 5)", 0, 5, 0)
        web_visits = st.slider("Web Visits per Month", 0, 20, 5)

    if st.button("Predict Segment", type="primary"):
        input_df = pd.DataFrame([{
            "Income": income,
            "Age": age,
            "Recency": recency,
            "Customer_Tenure_Days": tenure_days,
            "Total_Children": total_children,
            "Family_Size": family_size,
            "Total_Spending": total_spending,
            "Total_Purchases": total_purchases,
            "Total_Campaigns_Accepted": campaigns_accepted,
            "NumWebVisitsMonth": web_visits,
            "Education": education,
            "Marital_Status": marital_status,
        }])

        cluster = pipeline.predict(input_df)[0]

        st.success(f"### Predicted Segment: Cluster {cluster}")
        st.write(f"**Profile:** {CLUSTER_PERSONAS.get(cluster, 'No description available.')}")

        st.subheader("How this compares to the cluster average")
        st.dataframe(profiles.loc[[cluster]])

    st.divider()
    st.subheader("All Segment Profiles")
    st.dataframe(profiles)


if __name__ == "__main__":
    main()