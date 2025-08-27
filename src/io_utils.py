from __future__ import annotations
import os, json, time
import pandas as pd

RUN_TS = lambda: time.strftime("%Y%m%d-%H%M%S")

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all") # drop fully empty rows and columns
    return df

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    # identify whether the problem is classification or regression
    if (pd.api.types.is_numeric_dtype(y)) and (y.nunique() <= max(20, int(0.05*len(y)))):
        return "Classification"
    return "Regression"

def ensure_dirs():
    """
    This function confirm all necessary folders exist
    """
    for p in ["artefacts", "artefacts/eda_plots", "reports", "logs"]:
        os.makedirs(p, exist_ok=True)

def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_run_log(run_id: str, log: dict):
    write_json(log, f"logs/run_{run_id}.json")