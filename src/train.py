"""
End-to-end training script: clean data -> engineer features -> fit K-Means -> save pipeline.
"""
import pandas as pd
import joblib

from data_processing import preprocess_pipeline
from clustering import build_preprocessor, CLUSTERING_FEATURES
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

N_CLUSTERS = 5


def train_and_save(raw_path: str, processed_path: str, model_path: str) -> pd.DataFrame:
    # 1. Clean + engineer features
    df = preprocess_pipeline(raw_path, processed_path)

    # 2. Fit preprocessing + K-Means as one pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=42)),
    ])
    df["Cluster"] = pipeline.fit_predict(df[CLUSTERING_FEATURES])

    # 3. Save the fitted pipeline (preprocessing + model together — no train/serve skew)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    # 4. Save cluster-labeled data for profiling/reporting
    df.to_csv("data/processed/customer_segments.csv", index=False)

    return df


def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each cluster's average characteristics — turns raw cluster IDs into
    business-readable personas, which is the actual point of segmentation."""
    profile_cols = [
        "Age", "Income", "Total_Spending", "Total_Purchases",
        "Total_Children", "Family_Size", "Recency", "Total_Campaigns_Accepted",
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