from __future__ import annotations
import warnings
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

def run_automl_or_baseline(problem: str, preprocessor, X_train, y_train, X_test, y_test):
    """
    This function tries auto-sklearn if available; else run simple baselines.
    Returns: best_model, metrics(dict), model_name(str)
    """

    # Try Auto-sklearn 
    try:
        if problem == "Classification":
            model = Pipeline([
                ("pre", preprocessor),
                ("clf", AutoSklearnClassifier(time_left_for_this_task=180, per_run_time_limit=30))
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro"))
            }
            return model, metrics, "AutoSklearnClassifier"
        else:
            model = Pipeline([
                ("pre", preprocessor),
                ("reg", AutoSklearnRegressor(time_left_for_this_task=180,per_run_time_limit=30))
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            metrics = {"rmse": float(rmse)}
            return model, metrics, "AutoSklearnRegressor"
    except Exception as e:
        warnings.warn(f"Auto-sklearn unavailable or failed, using baselines. Reason: {e}")
    
    # Baselines
    if problem == "Classification":
        candidates = [
            ("LogisticRegression", LogisticRegression(max_iter=200)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=300, random_state=42))
        ]
        best = None

        for name, est in candidates:
            pipe = Pipeline([
                ("pre", preprocessor),
                ("est", est)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro"))
            }
            if (best is None) or (metrics["f1_macro"] > best[1]["f1_macro"]):
                best = (name, pipe, metrics)
        return best[1], best[2], best[0]
    else:
        candidates = [
            ("Ridge", Ridge(alpha=1.0)),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=300, random_state=42))
        ]
        best = None

        for name, est in candidates:
            pipe = Pipeline([("pre", preprocessor), ("est", est)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            metrics = {"rmse": float(rmse)}
            if (best is None) or (metrics["rmse"] < best[1]["rmse"]):
                best = (name, pipe, metrics)
        return best[1], best[2], best[0]

    