# Mall Customer Segmentation — KMeans Clustering

Machine Learning | Algonquin College

## Overview
Unsupervised machine learning app that segments mall customers into distinct groups based on their annual income and spending behaviour using KMeans clustering.

## Live Demo
[https://mall-clustering-anam.streamlit.app/]

## Dataset
`mall_customers.csv` — 200 customers, 5 features: Customer_ID, Gender, Age, Annual_Income, Spending_Score

## Features
- **Tab 1 — Data Overview:** Dataset shape, sample records, summary statistics, gender distribution
- **Tab 2 — Train & Evaluate:** Elbow plot, KMeans clustering, scatter plot with centroids, cluster distribution chart, cluster summary table
- **Tab 3 — Predict:** Enter a new customer's details and get their cluster assignment instantly

## Model
- Algorithm: KMeans with k-means++ initialisation
- Default clusters: k=5
- Feature sets: 2-feature (Annual Income + Spending Score) or 3-feature (Age + Annual Income + Spending Score)
- Preprocessing: StandardScaler normalisation
