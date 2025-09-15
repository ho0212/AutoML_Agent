from __future__ import annotations
import warnings, numpy as np
from typing import List, Optional
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

CLF_POOL = {
    "LogisticRegression": LogisticRegression(max_iter=300),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
}
REG_POOL = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
}

def _metric(problem, y_true, y_pred):
    if problem=="classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
        }
    else:
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}

def run_automl_or_baseline(problem: str, preprocessor, X_train, y_train, X_test, y_test,
                           strategy: str = "autosklearn",
                           candidates: Optional[List[str]] = None,
                           time_budget_sec: int = 180):
    """
    If strategy=='autosklearn', try it; otherwise use baselines limited by candidates list.
    """
    # Try autosklearn if requested
    if strategy == "autosklearn":
        try:
            if problem == "classification":
                from autosklearn.classification import AutoSklearnClassifier
                model = Pipeline([
                    ("pre", preprocessor),
                    ("clf", AutoSklearnClassifier(time_left_for_this_task=time_budget_sec,
                                                  per_run_time_limit=min(30, max(10, time_budget_sec//6))))
                ])
            else:
                from autosklearn.regression import AutoSklearnRegressor
                model = Pipeline([
                    ("pre", preprocessor),
                    ("reg", AutoSklearnRegressor(time_left_for_this_task=time_budget_sec,
                                                 per_run_time_limit=min(30, max(10, time_budget_sec//6))))
                ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return model, _metric(problem, y_test, y_pred), model.steps[-1][0]  # 'clf' or 'reg'
        except Exception as e:
            warnings.warn(f"Auto-sklearn unavailable or failed, falling back. Reason: {e}")

    # Baseline path
    pool = CLF_POOL if problem=="classification" else REG_POOL
    # Filter to known candidates
    chosen = []
    for name in (candidates or []):
        if name in pool:
            chosen.append((name, pool[name]))
    if not chosen:
        # default fallback
        chosen = [("RandomForestClassifier", pool.get("RandomForestClassifier"))] if problem=="classification" \
                 else [("RandomForestRegressor", pool.get("RandomForestRegressor"))]

    best = None  # (name, model, metrics)
    for name, est in chosen:
        pipe = Pipeline([("pre", preprocessor), ("est", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _metric(problem, y_test, y_pred)
        if best is None:
            best = (name, pipe, metrics)
        else:
            # Selection: for clf by f1_macro then accuracy; for reg by rmse lower is better
            if problem=="classification":
                cur_f1, best_f1 = metrics["f1_macro"], best[2]["f1_macro"]
                cur_acc, best_acc = metrics["accuracy"], best[2]["accuracy"]
                if (cur_f1 > best_f1) or (cur_f1 == best_f1 and cur_acc > best_acc):
                    best = (name, pipe, metrics)
            else:
                if metrics["rmse"] < best[2]["rmse"]:
                    best = (name, pipe, metrics)
    return best[1], best[2], best[0]
