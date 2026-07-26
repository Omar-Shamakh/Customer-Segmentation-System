"""
End-to-end training script: clean data -> engineer features -> fit PCA+K-Means -> save pipeline.
"""
import pandas as pd
import joblib

from data_processing import preprocess_pipeline
from clustering import build_full_pipeline, CLUSTERING_FEATURES


def train_and_save(raw_path: str, processed_path: str, model_path: str) -> pd.DataFrame:
    df = preprocess_pipeline(raw_path, processed_path)

    pipeline = build_full_pipeline()
    df["Cluster"] = pipeline.fit_predict(df[CLUSTERING_FEATURES])

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    df.to_csv("data/processed/customer_segments.csv", index=False)
    return df


def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [
        "Age", "Income", "Total_Spending", "NumWebPurchases",
        "NumStorePurchases", "NumWebVisitsMonth", "Recency", "Total_Campaigns_Accepted",
    ]
    summary = df.groupby("Cluster")[profile_cols].mean().round(1)
    summary["Count"] = df.groupby("Cluster").size()
    return summary


if __name__ == "__main__":
    df = train_and_save(
        raw_path="data/raw/customer_segmentation.csv",
        processed_path="data/processed/customer_segmentation_clean.csv",
        model_path="models/kmeans_pipeline.joblib",
    )

    profile = profile_clusters(df)
    print(profile)
    profile.to_csv("models/cluster_profiles.csv")