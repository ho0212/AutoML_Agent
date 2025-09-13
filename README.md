# Autonomous Data Science Assistant (AutoML Agent)

## Overview
This project demonstrates a **cutting-edge AutoML agent** that can take a raw dataset and autonomously perform:
- Exploratory Data Analysis (EDA) with plots
- Data preprocessing (imputation, scaling, encoding)
- Automated model selection & training
- Metric evaluation (classification & regression)
- AI-generated report narratives
- Agent planning with retries on errors

The result is a **self-driving data scientist assistant** that outputs a polished, human-readable report.

---

## Project Milestones
1. **M1** – Project scope defined  
2. **M2** – Skeleton pipeline (EDA → AutoML → Metrics, no LLM)  
3. **M3** – LLM Narrative added (reports include AI-written summary)  
4. **M4** – Agent Loop (LLM proposes plans, retries on error, logs decisions)  
5. **M5** – Demo polish (sample datasets, tidy artefacts, README, screenshots/GIF)  

---

##  Tech Stack
- **Python** (3.10+)  
- **scikit-learn** for preprocessing & models  
- **AutoML** (baseline models, optional Auto-sklearn if available)  
- **Gemini LLM (Google AI Studio)** for planning & narratives  
- **Matplotlib / Seaborn** for plots  
- **dotenv** for config management  

---

## Repository Structure

| Folder | Description |
|--------|-------------|
| `artefacts/` | Saved models, plots, metrics |
| `data/` | Example datasets (Titanic, Housing) |
| `docs/` | Project documents |
| `logs/` | Logs |
| `reports/` | Final reports (markdown) |
| `src/` | Source code |

---

## How to Run

1. Clone repo & create environment:
   ```bash
   conda create -n automl_agent python=3.10
   conda activate automl_agent
   pip install -r requirements.txt
2. Add a `.env` file with your API key:
   ```bash
   ENABLE_LLM_NARRATIVE=1
    GOOGLE_API_KEY=your_gemini_key
    GEMINI_MODEL=model name
3. Run on Titanic dataset (classification):
   ```bash
   python -m src.main --csv data/titanic.csv --target Survived --name Titanic
4. Run on Housing dataset (regression):
   ```bash
   python -m src.main --csv data/housing.csv --target median_house_value --name Housing
5. Reports are saved in `reports/` as Markdown files


