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