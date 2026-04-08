"""
model.py
--------
KMeans clustering logic for Mall Customer Segmentation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def compute_wcss(X_scaled: np.ndarray, k_range: range) -> list:
    """
    Compute Within-Cluster Sum of Squares for a range of k values (elbow method).
    """
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 5, max_iter: int = 300) -> KMeans:
    """
    Fit KMeans with k-means++ initialisation.
    """
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        max_iter=max_iter,
        random_state=42,
    )
    km.fit(X_scaled)
    return km


def get_cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Return per-cluster mean statistics.
    """
    df2 = df.copy()
    df2["Cluster"] = labels
    summary = df2.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].mean().round(2)
    summary["Count"] = df2.groupby("Cluster").size()
    return summary.reset_index()
