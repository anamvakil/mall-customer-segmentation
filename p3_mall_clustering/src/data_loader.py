"""
data_loader.py
--------------
Loads the Mall Customer dataset from data/mall_customers.csv.
"""

import pandas as pd
import os


def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join("data", "mall_customers.csv")
    df = pd.read_csv(path)
    return df
