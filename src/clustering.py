"""
Clustering pipeline: preprocessing (log-transform + scale), K-Means, and cluster evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Final curated feature set — chosen via systematic comparison against a 10-feature
# baseline and the tutorial's original set. This lean set scored highest on Silhouette
# (~0.28 at k=3) by avoiding low-signal, redundant features (Family_Size, Total_Children,
# Customer_Tenure_Days, Total_Purchases) that diluted K-Means distances.
CLUSTERING_FEATURES = [
    "Age", "Income", "Total_Spending",
    "NumWebPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Recency", "Total_Campaigns_Accepted",
]

# Right-skewed features that benefit from a log transform before scaling
SKEWED_FEATURES = ["Income", "Total_Spending"]

N_CLUSTERS = 3


def log_transform_skewed(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to skewed columns only, leave the rest untouched."""
    X = X.copy()
    for col in SKEWED_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


def build_preprocessor() -> Pipeline:
    """Log-transform skewed features, then standard-scale everything."""
    return Pipeline([
        ("log_transform", FunctionTransformer(log_transform_skewed, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])


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


def train_final_model(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> Pipeline:
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