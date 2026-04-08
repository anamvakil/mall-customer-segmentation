"""
preprocessor.py
---------------
Feature selection and scaling for Mall Customer Segmentation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def select_features(df: pd.DataFrame, feature_set: str = "2features") -> pd.DataFrame:
    """
    Return the feature columns used for clustering.

    feature_set options:
        '2features' -> Annual_Income, Spending_Score
        '3features' -> Age, Annual_Income, Spending_Score
    """
    if feature_set == "3features":
        cols = ["Age", "Annual_Income", "Spending_Score"]
    else:
        cols = ["Annual_Income", "Spending_Score"]
    return df[cols].copy()


def scale_features(X: pd.DataFrame):
    """
    Standardise features. Returns scaled array and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
