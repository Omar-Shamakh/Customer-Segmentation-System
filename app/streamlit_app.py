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
    0: "High income, high spending, but low campaign engagement — established value segment",
    1: "Highest income, highest spending, highest campaign response — VIP segment",
    2: "Lower income, low spending, low engagement — budget segment",
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
        total_spending = st.number_input("Total Spending ($, last 2 years)", 0, 3000, 500)
        total_purchases = st.slider("Total Purchases (all channels)", 0, 40, 10)
        recency = st.slider("Days Since Last Purchase", 0, 100, 30)

    with col2:
        num_web_purchases = st.slider("Web Purchases", 0, 30, 5)
        num_store_purchases = st.slider("Store Purchases", 0, 20, 5)
        web_visits = st.slider("Web Visits per Month", 0, 20, 5)
        campaigns_accepted = st.slider("Campaigns Accepted (past 5)", 0, 5, 0)

    if st.button("Predict Segment", type="primary"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "Total_Spending": total_spending,
            "Total_Purchases": total_purchases,
            "NumWebPurchases": num_web_purchases,
            "NumStorePurchases": num_store_purchases,
            "NumWebVisitsMonth": web_visits,
            "Recency": recency,
            "Total_Campaigns_Accepted": campaigns_accepted,
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