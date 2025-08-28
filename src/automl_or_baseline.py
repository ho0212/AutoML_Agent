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
    