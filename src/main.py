from __future__ import annotations
import os, argparse, joblib
from io_utils import ensure_dirs, load_csv, detect_problem_type, RUN_TS, write_json, save_run_log
from eda import quick_overview, plot_distributions
from preprocess import split_xy, build_transformer, make_splits
from automl_or_baseline import run_automl_or_baseline
from report_md import make_markdown

def main(csv_path: str, target: str, dataset_name: str|None=None):
    ensure_dirs()
    run_id = RUN_TS()
    log = {"run_id": run_id, "steps": []}

    df = load_csv(csv_path)
    assert target in df.columns, f"Target '{target}' not in columns"

    problem = detect_problem_type(df, target)
    log["problem"] = problem

    overview = quick_overview(df, target)
    plots = plot_distributions(df, target)

    X, y = split_xy(df, target)
    pre, num_cols, cat_cols = build_transformer(X)
    X_tr, X_te, y_tr, y_te = make_splits(X, y)

    model, metrics, model_name = run_automl_or_baseline(problem, pre, X_tr, y_tr, X_te, y_te)

    # save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, f"artifacts/best_model_{run_id}.pkl")
    write_json(metrics, f"artifacts/metrics_{run_id}.json")

    # write report
    dsname = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]
    md = make_markdown(dsname, problem, overview, plots, model_name, metrics)
    report_path = f"reports/{dsname}_{run_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    log["report"] = report_path
    save_run_log(run_id, log)
    print(f"Done. Report: {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--name", default=None, help="Optional dataset name for report")
    args = ap.parse_args()
    main(args.csv, args.target, args.name)
