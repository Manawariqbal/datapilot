# DataPilot

An open-source autonomous data scientist for tabular data.

## Current MVP

CSV -> profiling -> problem detection -> baseline ML -> TabPFN -> comparison -> report

The LLM/agent layer will be added after the deterministic data-science tools are working.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Planned architecture

```text
CSV
 |
Dataset Profiler
 |
Problem Detector
 |
EDA / Quality Tools
 |
Agent (Gemma + LangGraph)
 |
+---- Baseline ML
+---- TabPFN
+---- Statistical Tests
 |
Experiment Evaluator
 |
Evidence-backed Report
```
