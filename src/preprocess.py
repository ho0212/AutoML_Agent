from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def split_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def build_transformer(X: pd.DataFrame, *,
                      impute_numeric="median",
                      impute_categorical="most_frequent",
                      scale_numeric=True,
                      one_hot_encode=True):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    num_steps = [("impute", SimpleImputer(strategy=impute_numeric))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler(with_mean=False)))
    num_pipe = Pipeline(num_steps)

    if one_hot_encode:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy=impute_categorical)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ])
    else:
        # still impute but skip encoding
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy=impute_categorical)),
        ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols

def make_splits(X, y, test_size=0.2, random_state=42):
    strat = y if (pd.api.types.is_integer_dtype(y) or pd.api.types.is_categorical_dtype(y) or y.nunique() < 50) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
