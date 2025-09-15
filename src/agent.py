from __future__ import annotations
import os, json, re
from typing import Any, Dict
import google.generativeai as genai

def _first(x, default=None):
    if isinstance(x, list) and x:
        return x[0]
    return x if x is not None else default

def _to_bool(x, default=True):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true","1","yes","y"): return True
        if s in ("false","0","no","n"): return False
    return default

def _model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(name, generation_config={"temperature": 0.2})

PLAN_SCHEMA = {
    "preprocess": {
        "impute_numeric": ["median", "mean"],
        "impute_categorical": ["most_frequent"],
        "scale_numeric": [True, False],
        "one_hot_encode": [True, False]
    },
    "modeling": {
        "strategy": ["autosklearn", "baseline"],
        "candidates": ["LogisticRegression","RandomForestClassifier","Ridge","RandomForestRegressor"]
    },
    "evaluation": {
        "primary_metric": ["f1_macro","accuracy","rmse"],
        "cv_folds": [3,5]
    },
    "time_budget_sec": "integer (60-600)"
}

def _extract_json(text: str) -> Dict[str, Any]:
    # Accept fenced or inline JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("Planner did not return JSON")
    return json.loads(m.group(0))

def plan(overview: Dict[str,Any], problem: str) -> Dict[str,Any]:
    """
    Ask Gemini to produce a JSON plan constrained to PLAN_SCHEMA.
    """
    mdl = _model()
    sys = (
        "You are an ML planner. Output ONLY valid JSON matching the schema. "
        "Choose practical defaults when uncertain. Respect task type."
    )
    ctx = {
        "problem": problem,
        "n_rows": overview.get("n_rows"),
        "n_cols": overview.get("n_cols"),
        "missing_top": dict(sorted(overview.get("missing_perc",{}).items(), key=lambda kv: kv[1], reverse=True)[:5]),
        "dtypes": overview.get("dtypes", {}),
        "target_counts": overview.get("target_value_counts", {})
    }
    schema_hint = json.dumps(PLAN_SCHEMA, indent=2)
    prompt = (
        f"{sys}\n\n"
        f"SCHEMA (allowed values are examples, not exhaustive):\n```json\n{schema_hint}\n```\n\n"
        f"CONTEXT:\n```json\n{json.dumps(ctx, indent=2)}\n```\n\n"
        "Rules:\n"
        "- If classification: prefer primary_metric=f1_macro; if labels are balanced, accuracy is fine.\n"
        "- If regression: primary_metric=rmse.\n"
        "- time_budget_sec: 120-300 for demos.\n"
        "- If many categoricals: one_hot_encode=true. If many numerics: scale_numeric=true.\n"
        "- If Windows (likely no autosklearn): set strategy='baseline' and choose 2 candidates.\n"
        "Return ONLY JSON. No commentary."
    )
    resp = mdl.generate_content(prompt)
    js = _extract_json(getattr(resp, "text", "") or "")
    # Light validation + defaults
    pp = js.get("preprocess", {})
    md = js.get("modeling", {})
    ev = js.get("evaluation", {})
    js.setdefault("time_budget_sec", 180)

    pp["impute_numeric"] = _first(pp.get("impute_numeric"), "median")
    pp["impute_categorical"] = _first(pp.get("impute_categorical"), "most_frequent")
    pp["scale_numeric"] = _to_bool(pp.get("scale_numeric"), True)
    pp["one_hot_encode"] = _to_bool(pp.get("one_hot_encode"), True)


    md.setdefault("strategy", "autosklearn")
    
    if problem == "classification":
        allowed = {"LogisticRegression","RandomForestClassifier"}
        default_cands = ["LogisticRegression","RandomForestClassifier"]
        ev.setdefault("primary_metric", "f1_macro")
    else:
        allowed = {"Ridge","RandomForestRegressor"}
        default_cands = ["Ridge","RandomForestRegressor"]
        ev.setdefault("primary_metric", "rmse")

    cands = md.get("candidates") or default_cands
    cands = [c for c in (cands if isinstance(cands, list) else [cands]) if c in allowed] or default_cands
    seen=set(); md["candidates"] = [x for x in cands if not (x in seen or seen.add(x))][:2]
    
    try:
        ev["cv_folds"] = int(_first(ev.get("cv_folds"), 5))
    except Exception:
        ev["cv_folds"] = 5
    try:
        js["time_budget_sec"] = int(js.get("time_budget_sec", 180))
    except Exception:
        js["time_budget_sec"] = 180

    js["preprocess"], js["modeling"], js["evaluation"] = pp, md, ev
    return js

def repair(last_action: str, error_message: str, problem: str) -> Dict[str,Any]:
    """
    Ask Gemini for a minimal fix to the plan section that failed.
    Returns a tiny JSON patch e.g. {"preprocess":{...}} or {"modeling":{...}}.
    """
    mdl = _model()
    prompt = (
        "You are debugging a failing ML pipeline plan. Output ONLY JSON patch (subset keys).\n"
        f"FAILED_ACTION: {last_action}\nERROR:\n```{error_message}```\n"
        "Fix by adjusting parameters or choosing alternative model(s). If Windows, prefer baseline over autosklearn.\n"
        "Return ONLY JSON patch like {\"preprocess\":{...}} or {\"modeling\":{...}}."
    )
    resp = mdl.generate_content(prompt)
    return _extract_json(getattr(resp, "text", "") or "")
