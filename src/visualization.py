"""
PCA-based visualization of K-Means clusters.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_clusters(X_processed, labels, save_path: str = None) -> pd.DataFrame:
    """Reduce the processed (encoded+scaled) feature space to 2D with PCA and
    scatter-plot the clusters. Returns the 2D coordinates with cluster labels attached."""
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_processed)

    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels

    explained = pca.explained_variance_ratio_
    print(f"Explained variance — PC1: {explained[0]:.2%}, PC2: {explained[1]:.2%}, "
          f"Total: {explained.sum():.2%}")

    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == cluster_id]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_id}", alpha=0.6, s=40)

    plt.title("Customer Segments (PCA-reduced, 2D projection)")
    plt.xlabel(f"PC1 ({explained[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained[1]:.1%} variance)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()

    return plot_df


if __name__ == "__main__":
    from clustering import build_preprocessor, CLUSTERING_FEATURES
    from sklearn.cluster import KMeans

    df = pd.read_csv("data/processed/customer_segmentation_clean.csv")

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(df[CLUSTERING_FEATURES])

    kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_processed)

    plot_pca_clusters(X_processed, labels, save_path="models/pca_clusters.png")