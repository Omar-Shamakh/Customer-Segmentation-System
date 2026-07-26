"""
PCA-based visualization of K-Means clusters.
Clustering now happens directly in PCA-reduced space, so this plots that exact space —
no separate/duplicate PCA fit needed.
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_clusters_2d(X_reduced, labels, explained_variance_ratio, save_path: str = None) -> pd.DataFrame:
    """Scatter-plot clusters using pre-computed 2D coordinates (already PCA-reduced
    upstream, in the same space K-Means was fit on)."""
    plot_df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels

    print(f"Explained variance — PC1: {explained_variance_ratio[0]:.2%}, "
          f"PC2: {explained_variance_ratio[1]:.2%}, "
          f"Total: {explained_variance_ratio.sum():.2%}")

    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == cluster_id]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_id}", alpha=0.6, s=40)

    plt.title("Customer Segments (PCA-reduced, 2D projection)")
    plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()

    return plot_df


if __name__ == "__main__":
    from clustering import build_scaling_pipeline, CLUSTERING_FEATURES, N_CLUSTERS
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    df = pd.read_csv("data/processed/customer_segmentation_clean.csv")

    scaling_pipeline = build_scaling_pipeline()
    X_scaled = scaling_pipeline.fit_transform(df[CLUSTERING_FEATURES])

    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    plot_clusters_2d(X_reduced, labels, pca.explained_variance_ratio_, save_path="models/pca_clusters.png")