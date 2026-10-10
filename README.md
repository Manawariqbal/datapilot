# 🧭 DataPilot

**An open-source autonomous data scientist for tabular data.**

Upload a CSV, pick a target column, and DataPilot profiles your data, checks its quality, trains a baseline model, challenges it with [TabPFN](https://github.com/PriorLabs/TabPFN), compares the results, and gives you a full evidence-backed report, all inside one Streamlit app.

```
CSV -> profiling -> problem detection -> baseline ML -> TabPFN -> comparison -> report
```

> **Status:** MVP. The deterministic data-science tools come first; the LLM/agent layer will be added on top of them (see [Roadmap](#roadmap)).

---

## Features

- **Dataset profiling:** rows, columns, column types, unique values, missing values and duplicate rows.
- **Data-quality checks:** deterministic quality flags surfaced as warnings.
- **Exploratory data analysis:** target summary and the strongest correlations with the target.
- **Baseline model:** a standard ML model to set the bar.
- **TabPFN challenger:** an optional second model to compare against the baseline (can be switched off).
- **Model comparison:** side-by-side metrics, a metric chart, and a written comparison conclusion.
- **Full experiment report:** one page with every detail from the run, viewable in the app and downloadable.
- **Saved artifacts:** each run is saved to disk, with the report stored alongside it.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

---

## How to use the app

1. **Upload** a CSV file (numerical and/or categorical columns).
2. **Review** the dataset overview: preview, missing values, duplicates, column types.
3. **Choose the target column** you want to predict.
4. Optionally untick **Run TabPFN challenger** to run only the standard workflow.
5. Click **🚀 Run DataPilot experiment**.
6. Explore the results tabs:

| Tab | What you'll find |
| --- | --- |
| 📊 Overview | Problem type, target, models evaluated, performance at a glance, key findings |
| 🧹 Data quality | Quality flags, missing values, duplicate rows |
| 🔎 EDA insights | Target analysis and strongest signals |
| 🤖 Model comparison | Metric table, metric chart and comparison evidence |
| 🧪 Experiment details | Errors/warnings, raw results object, artifact location |
| 📄 **Full report** | The complete report, with download buttons |

Changing the uploaded file, the target, or the TabPFN option clears the previous results so you never see stale output.

---

## The full report

After every experiment, the **📄 Full report** tab shows everything DataPilot found in one scrollable page. The report contains:

1. **Executive summary:** dataset size, target, problem type, models evaluated, best model, number of quality flags and errors, and the comparison conclusion.
2. **Dataset overview:** row/column counts, duplicates, missing cells, and a per-column table (type, unique values, missing count and %).
3. **Data quality:** quality flags, missing values, duplicate records.
4. **Exploratory data analysis:** target summary and the strongest correlations with the target.
5. **Model results:** full metric table, best model, and comparison evidence.
6. **Errors and warnings** reported by the experiment engine.
7. **Saved artifacts:** where the run was written on disk.

You can get the report in three ways:

- **Read it in the app** in the 📄 Full report tab (the Markdown source is also available in an expander).
- **Download it** with **⬇️ Download report (.md)**. The raw results are available with **⬇️ Download raw results (.json)**.
- **Find it on disk:** every run also writes `report.md` into its artifact folder, next to the files saved by `save_run`.

The "best model" is chosen from the first available metric in this order: `roc_auc`, `f1_weighted`, `accuracy`, `r2`, `rmse`, `mae` (for `rmse`/`mae`, lower is better). If none of these metrics are present, the report simply omits the best-model line.

---

## Architecture

```
CSV
 |
Dataset Profiler
 |
Problem Detector
 |
EDA / Quality Tools
 |
Agent (Gemma + LangGraph)      <- planned
 |
+---- Baseline ML
+---- TabPFN
+---- Statistical Tests
 |
Experiment Evaluator
 |
Evidence-backed Report
```

## Project structure

```
datapilot/
├── app.py              # Streamlit app (UI + full report builder)
├── datapilot/          # Core package
│   ├── profiling.py    #   profile_dataframe(): dataset profiling
│   └── experiments.py  #   run_experiments(), save_run(): experiment engine
├── data/               # Sample / local datasets
├── tests/              # Additional tests
├── test_*.py           # Tests for the agent, evidence, experiments, feature analysis
├── requirements.txt
└── README.md
```

### What the app expects from `run_experiments`

`run_experiments(df, target, run_tabpfn_model=True)` returns a dictionary. The UI and the report read these keys and tolerate any that are missing:

| Key | Contents |
| --- | --- |
| `problem_type` | e.g. `binary_classification` |
| `target` | name of the target column |
| `models` | list of dicts, one per model: a name column (`model`, `model_name` or `name`) plus metrics such as `accuracy`, `f1_weighted`, `roc_auc` |
| `eda` | `quality_flags` (list), `target_summary` (dict), `correlations_with_target` (list of dicts) |
| `model_comparison` | dict with a `conclusion` string and any supporting evidence |
| `errors` | list of errors/warnings raised during the run |

`save_run(results)` writes the run's artifacts to a folder and returns its path; the app adds `report.md` to that folder.

---

## Running the tests

```bash
python -m pytest
```

---

## Roadmap

- [x] CSV upload, profiling and data-quality summary
- [x] EDA and target-signal analysis
- [x] Baseline model and TabPFN challenger
- [x] Model comparison
- [x] Full in-app report with downloads
- [ ] LLM/agent layer (Gemma + LangGraph) that plans and runs analyses
- [ ] Statistical tests as agent tools
- [ ] Experiment evaluator that cross-checks agent claims against evidence
- [ ] Additional export formats for the report (HTML / PDF)

---

## Notes

- Correlations show association, not causation. A low correlation does not mean a feature is useless.
- Model metrics are calculated by the experiment engine, not estimated by a language model.
- Input is currently limited to CSV files.
