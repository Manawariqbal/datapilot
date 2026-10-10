from datapilot.agent.llm import get_llm


def extract_tool_result(evidence: list[dict], tool_name: str) -> dict:
    """Return the result produced by a specific tool."""
    for item in evidence:
        if item.get("tool") == tool_name:
            result = item.get("result")
            return result if isinstance(result, dict) else {}
    return {}


def format_number(value, digits: int = 4) -> str:
    """Format a numeric value in Python, not in the LLM."""
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def build_dataset_findings(eda_result: dict) -> str:
    """Build dataset findings directly from EDA evidence."""
    lines = []
    numeric_summary = eda_result.get("numeric_summary") or {}

    if numeric_summary:
        lines.append("Numeric feature summaries:")
        for column, stats in numeric_summary.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {column}: mean={format_number(stats.get('mean'))}; "
                f"median={format_number(stats.get('median'))}; "
                f"std={format_number(stats.get('std'))}; "
                f"min={format_number(stats.get('min'))}; "
                f"max={format_number(stats.get('max'))}; "
                f"missing={stats.get('missing', 'Not available')}; "
                f"unique={stats.get('unique', 'Not available')}."
            )

    target_summary = eda_result.get("target_summary")
    if target_summary:
        lines.append(f"Target summary: {target_summary}")

    categorical_summary = eda_result.get("categorical_summary") or {}
    if categorical_summary:
        lines.append(f"Categorical summaries: {categorical_summary}")

    return "\n".join(lines) or (
        "Dataset findings are unavailable because EDA summary evidence is missing."
    )


def build_data_quality(eda_result: dict) -> str:
    """Build data-quality findings directly from EDA evidence."""
    quality_flags = eda_result.get("quality_flags")
    numeric_summary = eda_result.get("numeric_summary") or {}

    missing_counts = {
        column: stats.get("missing")
        for column, stats in numeric_summary.items()
        if isinstance(stats, dict) and stats.get("missing") is not None
    }

    lines = []
    if missing_counts:
        lines.append(f"Missing values by numeric column: {missing_counts}")
    if quality_flags is not None:
        lines.append(f"EDA quality flags: {quality_flags}")

    return "\n".join(lines) or (
        "Data-quality details are unavailable in the supplied EDA evidence."
    )


def build_model_comparison(
    baseline_result: dict,
    tabpfn_result: dict,
    model_evidence: dict,
) -> str:
    """Render model results and comparison from Python-generated evidence."""
    lines = []

    if baseline_result:
        lines.append(
            f"Baseline model: {baseline_result.get('model', 'Not specified')}; "
            f"problem type={baseline_result.get('problem_type', 'Not specified')}; "
            f"accuracy={format_number(baseline_result.get('accuracy'))}; "
            f"weighted F1={format_number(baseline_result.get('f1_weighted'))}; "
            f"R2={format_number(baseline_result.get('r2'))}; "
            f"MAE={format_number(baseline_result.get('mae'))}."
        )
    else:
        lines.append("Baseline model result is unavailable.")

    if tabpfn_result:
        if tabpfn_result.get("error"):
            lines.append(f"TabPFN failed: {tabpfn_result['error']}")
        else:
            lines.append(
                f"Challenger model: {tabpfn_result.get('model', 'Not specified')}; "
                f"problem type={tabpfn_result.get('problem_type', 'Not specified')}; "
                f"accuracy={format_number(tabpfn_result.get('accuracy'))}; "
                f"weighted F1={format_number(tabpfn_result.get('f1_weighted'))}; "
                f"R2={format_number(tabpfn_result.get('r2'))}; "
                f"MAE={format_number(tabpfn_result.get('mae'))}; "
                f"samples used={tabpfn_result.get('samples_used', 'Not specified')}."
            )
    else:
        lines.append("TabPFN result is unavailable.")

    if model_evidence.get("status") == "completed":
        lines.append(
            "Deterministic comparison: "
            f"baseline={model_evidence.get('baseline_model', 'Not specified')}; "
            f"challenger={model_evidence.get('challenger_model', 'Not specified')}; "
            f"baseline accuracy={format_number(model_evidence.get('baseline_accuracy'))}; "
            f"challenger accuracy={format_number(model_evidence.get('challenger_accuracy'))}; "
            f"accuracy winner={model_evidence.get('accuracy_winner', 'Not specified')}; "
            f"baseline weighted F1={format_number(model_evidence.get('baseline_f1_weighted'))}; "
            f"challenger weighted F1={format_number(model_evidence.get('challenger_f1_weighted'))}; "
            f"weighted F1 winner={model_evidence.get('f1_winner', 'Not specified')}; "
            f"conclusion={model_evidence.get('conclusion', 'Not specified')}."
        )
    else:
        lines.append(
            "A deterministic model-comparison summary is not available. "
            "Do not infer a winner or calculate differences."
        )

    return "\n".join(lines)


def build_feature_findings(feature_result: dict) -> str:
    """Render feature-analysis evidence directly from Python results."""
    if not feature_result:
        return "Feature-analysis evidence is unavailable."

    lines = [
        f"Method: {feature_result.get('method', 'Not specified')}; "
        f"target: {feature_result.get('target', 'Not specified')}."
    ]

    for item in feature_result.get("features", []):
        lines.append(
            f"- {item.get('feature', 'Unknown feature')}: "
            f"correlation={format_number(item.get('correlation'))}; "
            f"p-value={format_number(item.get('p_value'))}; "
            f"direction={item.get('direction', 'Not specified')}; "
            f"significance={item.get('significance', 'Not specified')}; "
            f"samples={item.get('samples', 'Not specified')}."
        )

    note = feature_result.get("interpretation_note")
    if note:
        lines.append(f"Interpretation note: {note}")

    return "\n".join(lines)


def generate_final_report(state):
    """Generate factual sections with Python; use the LLM for interpretation."""
    print()
    print("===== GENERATING FINAL REPORT =====")

    evidence = state.get("evidence") or []
    model_evidence = state.get("model_evidence") or {}
    target = state.get("target", "Not specified")
    problem_type = state.get("problem_type", "Not specified")

    baseline_result = extract_tool_result(evidence, "run_baseline")
    tabpfn_result = extract_tool_result(evidence, "run_tabpfn")
    eda_result = extract_tool_result(evidence, "run_eda")
    feature_result = (
        state.get("feature_evidence")
        or extract_tool_result(evidence, "run_feature_analysis")
    )

    # All factual/numeric sections below are generated by Python.
    dataset_findings = build_dataset_findings(eda_result)
    data_quality = build_data_quality(eda_result)
    model_comparison = build_model_comparison(
        baseline_result, tabpfn_result, model_evidence
    )
    feature_findings = build_feature_findings(feature_result)

    llm = get_llm()
    prompt = f"""
You are DataPilot's scientific interpretation assistant.

Python has already generated the factual sections below. Do not
reproduce or rewrite those sections, and do not introduce numbers.
Write only the two requested sections at the end.

Target: {target}
Problem type: {problem_type}

DATASET FINDINGS:
{dataset_findings}

DATA QUALITY:
{data_quality}

MODEL COMPARISON:
{model_comparison}

FEATURE ANALYSIS:
{feature_findings}

Rules:
- Interpret only the supplied evidence.
- Do not introduce numerical values or metrics.
- Correlation does not prove causation.
- If accuracy and weighted F1 have different winners, explain that
  the metrics do not identify the same model as winner.
- Do not claim one model is universally better based on these results.
- Recommendations must be future investigations, not completed work.
- If evidence is unavailable, say so rather than guessing.

Write exactly these sections:
## 4. Evidence-Based Conclusion
## 5. Recommended Next Investigation

Keep the response concise.
"""

    response = llm.invoke(prompt)
    interpretation = response.content.strip()

    final_report = f"""# DataPilot Report: {target} {str(problem_type).title()} Task

## 1. Dataset Findings

{dataset_findings}

## 2. Data Quality

{data_quality}

## 3. Model Comparison

{model_comparison}

## Feature Analysis Evidence

{feature_findings}

{interpretation}
"""

    print()
    print("===== FINAL REPORT GENERATED =====")
    print(final_report)

    return {
        "final_report": final_report,
        "model_evidence": model_evidence,
        "evidence": evidence,
        "completed_tools": state.get("completed_tools") or [],
        "hypothesis": state.get("hypothesis", ""),
        "reasoning": state.get("reasoning", ""),
        "next_action": state.get("next_action", "finish"),
    }
