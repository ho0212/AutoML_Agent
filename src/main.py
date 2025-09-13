from __future__ import annotations
import os, argparse, joblib, json
from dotenv import load_dotenv
load_dotenv()

from src.io_utils import ensure_dirs, load_csv, detect_problem_type, RUN_TS, write_json, save_run_log
from src.eda import quick_overview, plot_distribution
from src.preprocess import split_xy, build_transformer, make_splits
from src.automl_or_baseline import run_automl_or_baseline
from src.report_md import make_markdown
from src.agent import plan as agent_plan, repair as agent_repair

def maybe_make_narrative(dsname, problem, overview, metrics, model_name):
    if os.getenv("ENABLE_LLM_NARRATIVE","0") != "1":
        return None
    try:
        from src.llm_narrative import generate_narrative
        return generate_narrative(dsname, problem, overview, metrics, model_name)
    except Exception as e:
        print(f"[warn] LLM narrative disabled due to: {e}")
        return None

def main(csv_path: str, target: str, dataset_name: str|None=None):
    ensure_dirs()
    run_id = RUN_TS()
    log = {"run_id": run_id, "steps": []}

    df = load_csv(csv_path)
    assert target in df.columns, f"Target '{target}' not in columns"

    problem = detect_problem_type(df, target)
    overview = quick_overview(df, target)
    plots = plot_distribution(df, target)
    log["overview"] = {k: v for k,v in overview.items() if k in ["n_rows","n_cols","dtypes"]}

    # ==== Agent Planning ====
    try:
        pl = agent_plan(overview, problem)
        log["plan_initial"] = pl
    except Exception as e:
        # Always set a safe default plan so `pl` exists
        print(f"[plan] error: {e}")
        pl = {
            "preprocess": {
                "impute_numeric": "median",
                "impute_categorical": "most_frequent",
                "scale_numeric": True,
                "one_hot_encode": True
            },
            "modeling": {
                "strategy": "baseline",
                "candidates": [
                    "RandomForestClassifier" if problem == "classification" else "RandomForestRegressor",
                    "LogisticRegression" if problem == "classification" else "Ridge"
                ]
            },
            "evaluation": {
                "primary_metric": "f1_macro" if problem == "classification" else "rmse",
                "cv_folds": 5
            },
            "time_budget_sec": 180
        }
        log["plan_initial_error"] = str(e)
        log["plan_initial_fallback"] = pl

    # ---- Sanitize plan: task-correct & keep up to 2 candidates ----
    if problem == "classification":
        allowed = {"LogisticRegression", "RandomForestClassifier"}
        pl["evaluation"]["primary_metric"] = "f1_macro"
    else:
        allowed = {"Ridge", "RandomForestRegressor"}
        pl["evaluation"]["primary_metric"] = "rmse"

    cands = pl["modeling"].get("candidates", [])
    filtered = [c for c in cands if c in allowed]
    if not filtered:
        filtered = list(allowed)  # default for safety

    max_cands = int(os.getenv("AGENT_MAX_CANDIDATES", "2"))
    seen = set()
    pl["modeling"]["candidates"] = [x for x in filtered if not (x in seen or seen.add(x))][:max_cands]

    # ==== Execute with guardrails + one repair attempt per stage ====
    try:
        X, y = split_xy(df, target)
        pre, *_ = build_transformer(
            X,
            impute_numeric=pl["preprocess"]["impute_numeric"],
            impute_categorical=pl["preprocess"]["impute_categorical"],
            scale_numeric=pl["preprocess"]["scale_numeric"],
            one_hot_encode=pl["preprocess"]["one_hot_encode"],
        )
    except Exception as e:
        log["repair_preprocess_error"] = str(e)
        try:
            patch = agent_repair("preprocess", str(e), problem)
            log["repair_preprocess_patch"] = patch
            pp = {**pl["preprocess"], **patch.get("preprocess", {})}
            pre, *_ = build_transformer(
                X,
                impute_numeric=pp["impute_numeric"],
                impute_categorical=pp["impute_categorical"],
                scale_numeric=pp["scale_numeric"],
                one_hot_encode=pp["one_hot_encode"],
            )
            pl["preprocess"] = pp
        except Exception as e2:
            log["repair_preprocess_failed"] = str(e2)
            raise

    X_tr, X_te, y_tr, y_te = make_splits(X, y)

    try:
        model, metrics, model_name = run_automl_or_baseline(
            problem, pre, X_tr, y_tr, X_te, y_te,
            strategy=pl["modeling"].get("strategy","baseline"),
            candidates=pl["modeling"].get("candidates"),
            time_budget_sec=int(pl.get("time_budget_sec",180))
        )
    except Exception as e:
        log["repair_modeling_error"] = str(e)
        try:
            patch = agent_repair("modeling", str(e), problem)
            log["repair_modeling_patch"] = patch
            # Adjust strategy/candidates
            md = {**pl["modeling"], **patch.get("modeling", {})}
            model, metrics, model_name = run_automl_or_baseline(
                problem, pre, X_tr, y_tr, X_te, y_te,
                strategy=md.get("strategy","baseline"),
                candidates=md.get("candidates"),
                time_budget_sec=int(pl.get("time_budget_sec",180))
            )
            pl["modeling"] = md
        except Exception as e2:
            log["repair_modeling_failed"] = str(e2)
            raise

    # Save artifacts
    os.makedirs("artefacts", exist_ok=True)
    import joblib
    joblib.dump(model, f"artefacts/best_model_{run_id}.pkl")
    write_json(metrics, f"artefacts/metrics_{run_id}.json")

    dsname = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]
    narrative = maybe_make_narrative(dsname, problem, overview, metrics, model_name)

    # Write report (now with decisions)
    agent_decisions_md = "```json\n" + json.dumps({
        "plan": pl,
        "repairs": {k:v for k,v in log.items() if k.startswith("repair")}
    }, indent=2) + "\n```"

    md = make_markdown(dsname, problem, overview, plots, model_name, metrics, narrative=narrative)
    md += "\n\n## Agent Decisions\n" + agent_decisions_md + "\n"
    report_path = f"reports/{dsname}_{run_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    # Persist full run log
    log["problem"] = problem
    log["report"] = report_path
    save_run_log(run_id, log)
    print(f"Done. Report: {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    main(args.csv, args.target, args.name)
