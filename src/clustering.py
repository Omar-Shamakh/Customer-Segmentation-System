"""
Clustering pipeline: preprocessing (log-transform + robust-scale) -> PCA -> K-Means, plus evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Final curated feature set — arrived at via systematic comparison of feature subsets,
# scalers, and PCA-vs-raw clustering spaces. Total_Purchases added back in after
# confirming it improves separation once combined with RobustScaler + PCA.
CLUSTERING_FEATURES = [
    "Age", "Income", "Total_Spending", "Total_Purchases",
    "NumWebPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Recency", "Total_Campaigns_Accepted",
]
SKEWED_FEATURES = ["Income", "Total_Spending", "Total_Purchases"]

# RobustScaler (median/IQR-based) outperformed StandardScaler here — this dataset has
# genuine extreme-spender outliers that distort mean/std-based scaling.
# Clustering on 2 PCA components (not raw scaled features) removes correlated noise
# before K-Means ever sees the data.
N_PCA_COMPONENTS = 2
N_CLUSTERS = 3


def log_transform_skewed(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in SKEWED_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


def build_scaling_pipeline() -> Pipeline:
    return Pipeline([
        ("log_transform", FunctionTransformer(log_transform_skewed, feature_names_out="one-to-one")),
        ("scaler", RobustScaler()),
    ])


def build_full_pipeline(n_clusters: int = N_CLUSTERS, n_components: int = N_PCA_COMPONENTS) -> Pipeline:
    return Pipeline([
        ("scaling", build_scaling_pipeline()),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("kmeans", KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)),
    ])


def find_optimal_k(X_reduced: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_reduced)
        results.append({
            "k": k,
            "wcss": km.inertia_,
            "silhouette": silhouette_score(X_reduced, labels),
        })
    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame) -> Pipeline:
    pipeline = build_full_pipeline()
    pipeline.fit(df[CLUSTERING_FEATURES])
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/customer_segmentation_clean.csv")

    scaling_pipeline = build_scaling_pipeline()
    X_scaled = scaling_pipeline.fit_transform(df[CLUSTERING_FEATURES])

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    eval_df = find_optimal_k(X_reduced)
    print(eval_df)
    eval_df.to_csv("data/processed/cluster_evaluation.csv", index=False)