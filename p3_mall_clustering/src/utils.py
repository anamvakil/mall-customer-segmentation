"""
utils.py
--------
Chart helpers for Mall Customer Segmentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_elbow(wcss: list, k_range: range) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), wcss, marker="o", color="steelblue", linewidth=2)
    ax.set_title("Elbow Plot — Optimal Number of Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS Score")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_clusters_2d(df: pd.DataFrame, labels: np.ndarray, centers: np.ndarray) -> plt.Figure:
    """
    Scatter plot of Annual_Income vs Spending_Score coloured by cluster.
    """
    df2 = df.copy()
    df2["Cluster"] = labels.astype(str)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df2, x="Annual_Income", y="Spending_Score",
        hue="Cluster", palette="colorblind", s=80, ax=ax,
    )
    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=250, marker="X",
               zorder=5, label="Centroids")
    ax.set_title("Customer Segments — Annual Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1–100)")
    ax.legend(title="Cluster")
    plt.tight_layout()
    return fig


def plot_clusters_3d_as_pairs(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """
    Pair of 2D scatter plots when 3 features are used.
    """
    df2 = df.copy()
    df2["Cluster"] = labels.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df2, x="Annual_Income", y="Spending_Score",
                    hue="Cluster", palette="colorblind", ax=axes[0])
    axes[0].set_title("Annual Income vs Spending Score")

    sns.scatterplot(data=df2, x="Age", y="Spending_Score",
                    hue="Cluster", palette="colorblind", ax=axes[1])
    axes[1].set_title("Age vs Spending Score")

    plt.tight_layout()
    return fig


def plot_cluster_distribution(labels: np.ndarray) -> plt.Figure:
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Cluster {i}" for i in unique], counts, color="steelblue", alpha=0.8)
    ax.set_title("Observations per Cluster")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return fig
