# Autonomous Data Science Assistant (AutoML Agent)

## Project Goal
Build a Python agent that accepts a CSV + target column, performs EDA $\rightarrow$ cleaning $\rightarrow$ AutoML $\rightarrow$ evaluation, and emits a markdown report with plots and an LLM-generated narrative.
## Project Input & Output
- **Input:** A `CSV` file, the target column, and the problem type (optional)
- **Output:** 
    1. Report
    2. Model
    3. EDA plots
    4. Metrics
    5. Logs

## Capabilities (MVP)
- Automatically detect problem types (Classification/Regression)
- EDA (stats, missing values, plots)
- Data Cleansing (data imputation, one-hot encoding, scale)
- Auto-sklearn
- Auto model selection based on Accuracy/F1 (classification) and RMSE (regression)
- Reports with plots + LLM narrative + feature importance

## Guardrails & Permissions
- No **network calls** from generated code (disable `requests`, `subprocess`, `os.system`)
- **Filesystem Sandbox:** Only write to assigned folders (e.g. `reports/`, `artefacts/`, `logs/`)
- **Execution Policy:** 
    - Run LLM-generated code in a try/except wrapper
    - On error, pass traceback back to LLM once
    - If still failing, fall back to trusted library path.
    - Resouce caps including row cap (e.g., 100k) and AutoML time cap (e.g., 300s default; 60s in tests).

## Success Criteria & Metrics

- **Functional:** On two public datasets (one classification, one regression), the agent completes EDA $\rightarrow$ report in < 8 minutes on a laptop.
- **Quality**:
    - **Titanic Dataset (Classification):** Accuracy $\geq$ 0.75 or F1-Score $\geq$ 0.75 (baseline sanity)
    - **California Housing Dataset (Regression):** RMSE within 15% of sklearn baselines
- **Reliability:** 
    - 90% of runs complete without manual fixes 
    - Report files always produced
- **Explainability:** 
    - Report includes top features or coeeficients
    - LLM narrative references actual numbers

## Evaluation Plan
- **Datasets:**
    - **Classification:** Titanic Dataset
    - **Regression:** California Housing Dataset
- **Tests:**
    1. Small synthetic `CSV` file (100 rows) for speed + plumbing
    2. Titanic full flow
    3. California Housing full flow
    4. Edge cases:
        1. All numeric
        2. Many categoricals (high cardinality)
        3. Missing target rows should be dropped with a warning
-  **Keeping Artefacts:** metric.json, confusion matrix (classification), residual plot (regression)

## Milestones
1. M1: The Completion of Project Scope
2. M2: Skeleton Pipeline (No LLM): CLI/Notebook that runs EDA $\rightarrow$ AutoML $\rightarrow$ Metrics
3. M3: LLM Narrative (Report sections generated from real stats)
4. M4: Agent Loop (LLM proposes steps + calls tools; Retries on error)
5. M5: Demo Polish (Two sample dataset runs, Tidy Artefacts, `README`, screenshots/GIF)
