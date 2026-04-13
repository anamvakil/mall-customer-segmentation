# Mall Customer Segmentation

**CST2216 — Modularising and Deploying ML Code | Algonquin College**

A deployed unsupervised machine learning web application that segments mall customers into distinct groups based on annual income and spending behaviour using KMeans clustering. Built as part of a modular ML deployment project, converted from a Jupyter notebook into a production-ready Streamlit app.

🔗 **Live App:** [mall-clustering-anam.streamlit.app](https://mall-clustering-anam.streamlit.app)

---

## Overview

This app applies KMeans clustering to 200 mall customers with no labels — patterns are discovered purely from data structure. The optimal number of clusters (k=5) was selected using an elbow plot. The interface lets users explore the dataset, run the clustering pipeline with adjustable parameters and assign new customers to a segment in real time.

| Result | Value |
|---|---|
| Algorithm | KMeans |
| Optimal Clusters (k) | 5 |
| Dataset Size | 200 customers |
| Features Used | Annual Income, Spending Score |

---

## Project Structure

```
mall-customer-segmentation/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Dependencies (no pinned versions)
├── data/
│   └── mall_customers.csv  # 200 customer records
└── src/
    ├── __init__.py
    ├── data_loader.py      # Loads and validates the dataset
    ├── preprocessor.py     # Feature scaling and selection
    ├── model.py            # KMeans clustering and elbow plot logic
    └── utils.py            # Shared helper functions
```

---

## App Features

**Tab 1 — Data Overview**
- Dataset shape, sample rows and summary statistics
- Feature distribution visualisations

**Tab 2 — Train & Evaluate**
- Adjustable hyperparameters (number of clusters k, max iterations, feature set)
- Elbow plot to visualise optimal k selection
- Two visualisation modes: 2-feature scatter and 3-feature view
- Cluster scatter plot with centroids marked

**Tab 3 — Predict**
- Input sliders for annual income and spending score
- Instant cluster assignment for a new customer
- Per-customer segment label returned from the trained model

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | UI framework and cloud deployment |
| scikit-learn | KMeans clustering |
| pandas / NumPy | Data loading and preprocessing |
| Matplotlib / Seaborn | Scatter plots, elbow plot, centroid visualisation |
| GitHub | Version control and auto-deploy trigger |

---

## Running Locally

```bash
git clone https://github.com/anamvakil/mall-customer-segmentation
cd mall-customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.9 or higher.

---

## Key Technical Notes

- **Unsupervised task** — no labels used; clusters are discovered from data structure alone
- **k=5 selected via elbow plot** — inertia curve flattens at k=5 across income and spending dimensions
- **No pinned versions** in `requirements.txt` — compatible with Streamlit Cloud's Python 3.14 runtime
- **Pandas 2.x compatible** — all `inplace=True` calls replaced with explicit assignment syntax
- **data/ folder committed to repo** — required for Streamlit Cloud to locate the CSV at runtime
- `plt.close(fig)` called after every `st.pyplot()` call to prevent matplotlib memory accumulation on the cloud

---

## Dataset

`mall_customers.csv` — 200 rows with annual income (k$) and spending score (1–100) features. No missing values. Unsupervised task; no target label column.

---

## Cluster Results

Five distinct customer segments identified across income and spending dimensions:

| Cluster | Profile |
|---|---|
| 0 | Low income, low spending |
| 1 | Low income, high spending |
| 2 | Medium income, medium spending |
| 3 | High income, low spending |
| 4 | High income, high spending |

Centroids and per-customer cluster assignments are visible in the live app.

---

## Author

**Anam Vakil**  
BISI Graduate Certificate — Algonquin College  
[github.com/anamvakil](https://github.com/anamvakil)
