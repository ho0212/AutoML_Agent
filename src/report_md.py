from __future__ import annotations
from tabulate import tabulate
import os, json

def make_markdown(dataset_name: str, problem: str, overview: dict, plot_paths: list[str], model_name: str, metrics: dict) -> str:
    lines = [] #content
    lines += [f"# AutoML Agent Report - {dataset_name}"]
    lines += ["## Overview",
              f"- Problem Type: **{problem}**",
              f"- Rows: {overview["n_rows"]}, Cols: {overview["n_cols"]}",
              f"- Target distribution (top): {str(dict(list(overview.get('target_value_counts', {}).items())[:5]))}",
              ""
              ]
    lines += ["## Missingness (top 10)",
              "```",
              json.dumps({k:v for k,v in sorted(overview['missing_perc'].items(), key=lambda kv: kv[1], reverse=True)[:10]}, indent=2),
              "```",
              ""]
    if plot_paths:
        lines += ["## EDA Plots", ""]
        for p in plot_paths[:12]:
            rel = os.path.relpath(p)
            lines += [f"![{os.path.basename(p)}]({rel})"]
        lines += [""]
    
    lines += ["## Model & Metrics",
              f"- Selected model: **{model_name}**",
              "",
              "```\n" + tabulate([[k, v] for k,v in metrics.items()], headers=["metric","value"]) + "\n```",
              ""]
    lines += ["## Notes",
              "- This MVP report is generated without LLM. Narrative will be added in Step 3.",
              ""]
    return "\n".join(lines)