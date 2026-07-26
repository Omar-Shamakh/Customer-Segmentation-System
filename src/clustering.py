"""
Clustering pipeline: preprocessing (encode + scale), K-Means, and cluster evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Features used for clustering — deliberately curated, not "throw every column in"
NUMERIC_FEATURES = [
    "Income", "Age", "Recency", "Customer_Tenure_Days",
    "Total_Children", "Family_Size", "Total_Spending",
    "Total_Purchases", "Total_Campaigns_Accepted", "NumWebVisitsMonth",
]
CATEGORICAL_FEATURES = ["Education", "Marital_Status"]
CLUSTERING_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, standard-scale numerics. Required before K-Means,
    since it's distance-based and sensitive to feature scale/units."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def find_optimal_k(X_processed: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """Compute WCSS (Elbow Method) and Silhouette Score for a range of k values."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_processed)
        results.append({
            "k": k,
            "wcss": km.inertia_,
            "silhouette": silhouette_score(X_processed, labels),
        })
    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame, n_clusters: int) -> Pipeline:
    """Fit the full preprocessing + K-Means pipeline as a single sklearn Pipeline object,
    so preprocessing and model are saved/loaded together — no train/serve skew."""
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("kmeans", KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)),
    ])
    pipeline.fit(df[CLUSTERING_FEATURES])
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/customer_segmentation_clean.csv")

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(df[CLUSTERING_FEATURES])

    eval_df = find_optimal_k(X_processed)
    print(eval_df)
    eval_df.to_csv("data/processed/cluster_evaluation.csv", index=False)