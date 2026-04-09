"""
app.py
------
Streamlit app — Mall Customer Segmentation via KMeans Clustering
Tabs: Data Overview | Train & Evaluate | Predict
Dataset loads automatically from data/mall_customers.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.preprocessor import select_features, scale_features
from src.model import compute_wcss, train_kmeans, get_cluster_summary
from src.utils import (
    plot_elbow, plot_clusters_2d, plot_clusters_3d_as_pairs,
    plot_cluster_distribution,
)

DATA_PATH = os.path.join("data", "mall_customers.csv")

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon=None,
    layout="wide",
)

st.title("Mall Customer Segmentation — KMeans Clustering")
st.caption("Machine Learning | Algonquin College")
st.markdown("---")


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


try:
    raw_df = load_raw_data()
except FileNotFoundError:
    st.error("Dataset not found at data/mall_customers.csv. Please ensure the file is committed to the repository.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Data Overview", "Train & Evaluate", "Predict"])


# ── TAB 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")
    st.write("Mall Customer dataset — 200 customers with demographic and spending features.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", raw_df.shape[0])
    col2.metric("Features", raw_df.shape[1] - 1)
    col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

    st.subheader("Sample Records")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.subheader("Gender Distribution")
    gender_counts = raw_df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    st.dataframe(gender_counts, use_container_width=True)


# ── TAB 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.header("Train & Evaluate KMeans Clustering")
    st.write("Configure the clustering parameters below and run the pipeline.")

    with st.expander("Hyperparameter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            feature_set = st.selectbox(
                "Feature set",
                ["2features", "3features"],
                format_func=lambda x: "Annual Income + Spending Score" if x == "2features"
                else "Age + Annual Income + Spending Score",
            )
            n_clusters = st.slider("Number of clusters (k)", 2, 10, 5)
        with col2:
            max_iter = st.slider("Max iterations", 100, 500, 300, step=50)
            k_max = st.slider("Elbow plot — max k to evaluate", 5, 15, 10)

    if st.button("Run Clustering Pipeline"):
        try:
            with st.spinner("Preprocessing data..."):
                X = select_features(raw_df, feature_set)
                X_scaled, scaler = scale_features(X)

            with st.spinner("Computing elbow curve..."):
                k_range = range(1, k_max + 1)
                wcss = compute_wcss(X_scaled, k_range)

            st.subheader("Elbow Plot")
            st.caption("Look for the 'elbow' — the point where WCSS stops dropping sharply.")
            try:
                fig_elbow = plot_elbow(wcss, k_range)
                st.pyplot(fig_elbow)
                plt.close(fig_elbow)
            except Exception as e:
                st.warning(f"Could not render elbow plot: {e}")

            with st.spinner(f"Training KMeans with k={n_clusters}..."):
                km = train_kmeans(X_scaled, n_clusters=n_clusters, max_iter=max_iter)
                labels = km.labels_
                raw_df_copy = raw_df.copy()
                raw_df_copy["Cluster"] = labels

            st.success(f"Clustering complete — {n_clusters} clusters formed.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters", n_clusters)
            m2.metric("Inertia (WCSS)", f"{km.inertia_:.1f}")
            m3.metric("Iterations to Converge", km.n_iter_)

            st.subheader("Cluster Scatter Plot")
            try:
                if feature_set == "2features":
                    centers_orig = scaler.inverse_transform(km.cluster_centers_)
                    fig_scatter = plot_clusters_2d(raw_df_copy, labels, centers_orig)
                    st.pyplot(fig_scatter)
                    plt.close(fig_scatter)
                else:
                    fig_pairs = plot_clusters_3d_as_pairs(raw_df_copy, labels)
                    st.pyplot(fig_pairs)
                    plt.close(fig_pairs)
            except Exception as e:
                st.warning(f"Could not render scatter plot: {e}")

            st.subheader("Observations per Cluster")
            try:
                fig_dist = plot_cluster_distribution(labels)
                st.pyplot(fig_dist)
                plt.close(fig_dist)
            except Exception as e:
                st.warning(f"Could not render distribution chart: {e}")

            st.subheader("Cluster Summary — Mean Statistics")
            summary = get_cluster_summary(raw_df, labels)
            st.dataframe(summary, use_container_width=True)

            st.session_state["cluster_state"] = {
                "model": km,
                "scaler": scaler,
                "feature_set": feature_set,
                "n_clusters": n_clusters,
            }

        except Exception as e:
            st.error(f"Pipeline error: {e}")


# ── TAB 3 ──────────────────────────────────────────────────────────────────
with tab3:
    st.header("Predict Cluster for a New Customer")

    if "cluster_state" not in st.session_state:
        st.info("Run the clustering pipeline first in the Train & Evaluate tab.")
    else:
        cs = st.session_state["cluster_state"]
        st.write("Enter the customer's details below.")

        col1, col2 = st.columns(2)
        with col1:
            annual_income = st.slider("Annual Income (k$)", 15, 140, 60)
            spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)
        with col2:
            age = st.slider("Age", 18, 70, 35)

        if st.button("Predict Cluster"):
            try:
                if cs["feature_set"] == "2features":
                    input_data = pd.DataFrame([[annual_income, spending_score]],
                                              columns=["Annual_Income", "Spending_Score"])
                else:
                    input_data = pd.DataFrame([[age, annual_income, spending_score]],
                                              columns=["Age", "Annual_Income", "Spending_Score"])

                input_scaled = cs["scaler"].transform(input_data)
                cluster = cs["model"].predict(input_scaled)[0]

                st.markdown("---")
                st.success(f"This customer belongs to **Cluster {cluster}**")
                st.caption(
                    f"Model trained with k={cs['n_clusters']} clusters using "
                    f"{'Annual Income + Spending Score' if cs['feature_set'] == '2features' else 'Age + Annual Income + Spending Score'}."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
