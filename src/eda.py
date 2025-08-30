from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = "artefacts/eda_plots"

def quick_overview(df: pd.DataFrame, target: str) -> dict:
    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_percentage": {c: float(df[c].isna().mean()) for c in df.columns},
        "target_value_counts": df[target].value_counts(dropna=False).to_dict() if target in df.columns else {}
    }

    return info

def plot_distribution(df: pd.DataFrame, target: str) -> list[str]:
    paths = []
    num_cols = [c for c in df.columns if (c != target) and (pd.api.types.is_numeric_dtype(df[c]))]
    cat_cols = [c for c in df.columns if (c != target) and (not pd.api.types.is_numeric_dtype(df[c]))]

    # Numeric hists
    for c in num_cols[:20]: # set a cap ensuring processing speed
        plt.figure()
        df[c].hist(bins=30)
        plt.title(f"Histogram for {c} Column")

        out = os.path.join(PLOT_DIR, f"hist_{c}.png") # set path for storing chart
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths.append(out)
    
    # Categorical bars
    for c in cat_cols[:20]: # set a cap ensuring processing speed
        plt.figure()
        df[c].value_counts(dropna=False).head(30).plot(kind="bar")
        plt.title(f"Bar Chart for {c} Column")

        out = os.path.join(PLOT_DIR, f"bar_{c}.png") # set path for storing chart
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths.append(out)
    
    # Correlation heatmap (numeric only)
    num_df = df[num_cols] # extract numeric columns

    if num_df.shape[1] >= 2: # generate heatmap if df has more than 2 columns
        plt.figure(figsize=(6,5))
        sns.heatmap(num_df.corr(numeric_only=True), annot=False)

        out = os.path.join(PLOT_DIR, "corr_heatmap.png") # set path for storing chart
        plt.title("Correlation heatmap")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths.append(out)
    
    return paths