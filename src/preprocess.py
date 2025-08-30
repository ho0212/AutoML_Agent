from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def split_train_target(df: pd.DataFrame, target: str):
    """
    Split training data and target column
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def build_transformer(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])] # numeric columns
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])] # categorical columns

    # numeric pipeline
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler(with_mean=False))
    ])

    # categorical pipeline
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # transformer
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return pre, num_cols, cat_cols

def make_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()<50 else None)