
from datapilot.agent.llm import get_llm


def extract_tool_result(
    evidence: list[dict],
    tool_name: str,
):
    """
    Extract the result produced by a specific tool.
    """

    for item in evidence:

        if item.get("tool") == tool_name:

            return (
                item.get("result")
                or {}
            )

    return {}


def generate_final_report(state):

    print()
    print("===== GENERATING FINAL REPORT =====")

    # ------------------------------------------------------------
    # READ STATE
    # ------------------------------------------------------------

    evidence = (
        state.get("evidence")
        or []
    )

    model_evidence = (
        state.get("model_evidence")
        or {}
    )

    target = state.get(
        "target"
    )

    problem_type = state.get(
        "problem_type"
    )

    # ------------------------------------------------------------
    # EXTRACT DETERMINISTIC TOOL RESULTS
    # ------------------------------------------------------------

    baseline_result = extract_tool_result(
        evidence,
        "run_baseline",
    )

    tabpfn_result = extract_tool_result(
        evidence,
        "run_tabpfn",
    )

    eda_result = extract_tool_result(
        evidence,
        "run_eda",
    )

    feature_result = extract_tool_result(
        evidence,
        "run_feature_analysis",
    )

    # ------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------

    print()
    print("===== REPORT STATE DEBUG =====")

    print()
    print("Evidence count:")
    print(
        len(evidence)
    )

    print()
    print("Baseline result:")
    print(
        baseline_result
    )

    print()
    print("TabPFN result:")
    print(
        tabpfn_result
    )

    print()
    print("Model evidence:")
    print(
        model_evidence
    )

    print()
    print("==============================")

    # ------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------

    llm = get_llm()

    prompt = f"""
You are the final scientific reporting agent for DataPilot.

Your job is to write a scientific interpretation of the
deterministic evidence produced by Python tools.

Python is responsible for calculating numerical facts.

You are responsible for interpreting those facts.

============================================================
STRICT RULES
============================================================

1. NEVER invent a number.

2. NEVER change a number.

3. NEVER calculate a new metric yourself.

4. NEVER claim two values are identical unless the supplied
   values are actually identical.

5. NEVER invent experiments.

6. NEVER invent statistical results.

7. NEVER invent feature importance.

8. NEVER claim correlation proves causation.

9. Only describe capabilities that are explicitly represented
   by the supplied tool evidence.

10. Recommendations must be future investigations and must
    not be presented as completed experiments.

============================================================
DATASET
============================================================

Target:
{target}

Problem type:
{problem_type}

============================================================
EDA EVIDENCE
============================================================

{eda_result}

============================================================
BASELINE MODEL EVIDENCE
============================================================

{baseline_result}

============================================================
TABPFN MODEL EVIDENCE
============================================================

{tabpfn_result}

============================================================
MODEL COMPARISON EVIDENCE
============================================================

{model_evidence}

============================================================
FEATURE ANALYSIS EVIDENCE
============================================================

{feature_result}

============================================================
FULL TOOL EVIDENCE
============================================================

{evidence}

============================================================
REPORT REQUIREMENTS
============================================================

Write a concise scientific report using exactly these
sections:

1. Dataset Findings
2. Data Quality
3. Model Comparison
4. Evidence-Based Conclusion
5. Recommended Next Investigation

============================================================
MODEL COMPARISON RULES
============================================================

If MODEL COMPARISON EVIDENCE is available:

- Report the baseline model.
- Report the challenger model.
- Report their actual accuracy values.
- Report their actual F1 values when available.
- Report the winner fields when available.
- If accuracy and F1 have different winners, explicitly
  explain that they disagree.
- If the metrics are close, describe them as close.
- Do NOT call different values identical.

If MODEL COMPARISON EVIDENCE is NOT available:

- Use the individual BASELINE MODEL EVIDENCE and
  TABPFN MODEL EVIDENCE.
- Clearly state that the individual model results are
  available but a deterministic model-comparison summary
  is not yet available.
- Do NOT invent a winner.
- Do NOT calculate the difference yourself.

============================================================
SCIENTIFIC INTERPRETATION
============================================================

For feature analysis:

- Use the supplied correlations and p-values.
- Statistical significance does not imply causation.
- Do not describe a feature as a causal driver.

For recommendations:

- Recommend only investigations that have not already been
  performed.
- Clearly label them as future investigations.

Do not include any information that is not supported by
the supplied evidence.
"""

    response = llm.invoke(
        prompt
    )

    final_report = response.content.strip()

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    print()
    print("===== FINAL REPORT GENERATED =====")

    print(
        final_report
    )

    # ------------------------------------------------------------
    # RETURN STATE
    # ------------------------------------------------------------

    return {
        "final_report": final_report,

        "model_evidence": model_evidence,

        "evidence": evidence,

        "completed_tools": (
            state.get(
                "completed_tools"
            )
            or []
        ),

        "hypothesis": state.get(
            "hypothesis",
            ""
        ),

        "reasoning": state.get(
            "reasoning",
            ""
        ),

        "next_action": state.get(
            "next_action",
            "finish"
        ),
    }
