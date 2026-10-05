from datapilot.agent.llm import get_llm


def generate_final_report(state):

    print()
    print("===== GENERATING FINAL REPORT =====")

    # ------------------------------------------------------------
    # READ STATE
    # ------------------------------------------------------------

    evidence = state.get("evidence") or []

    model_evidence = state.get("model_evidence") or {}

    target = state.get("target")

    problem_type = state.get("problem_type")

    # ------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------

    print()
    print("===== REPORT STATE DEBUG =====")

    print("Model evidence received by report:")
    print(model_evidence)

    print()
    print("Evidence count:")
    print(len(evidence))

    print()
    print("==============================")

    # ------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------

    llm = get_llm()

    prompt = f"""
You are the final scientific reporting agent for DataPilot.

Analyze ONLY the evidence provided below.

Do not invent:

- model metrics
- statistical results
- feature importance
- experiments
- causal relationships
- model comparisons
- tool capabilities

If information is not present in the evidence,
do not claim it.

============================================================
DATASET
============================================================

Target:
{target}

Problem type:
{problem_type}

============================================================
TOOL EVIDENCE
============================================================

{evidence}

============================================================
MODEL COMPARISON EVIDENCE
============================================================

{model_evidence}

============================================================
REPORT REQUIREMENTS
============================================================

Write a concise scientific report using exactly these sections:

1. Dataset Findings
2. Data Quality
3. Model Comparison
4. Evidence-Based Conclusion
5. Recommended Next Investigation

IMPORTANT:

- Use the actual numerical metrics from MODEL COMPARISON EVIDENCE.
- If model comparison evidence exists, report it.
- If accuracy and F1 disagree, explicitly mention that.
- If the models are very similar, say so.
- Do not say "no model comparison was performed"
  when model comparison evidence is provided.
- Do not claim TabPFN performed EDA.
- Do not claim TabPFN performed outlier detection.
- Do not claim TabPFN calculated feature importance.
- Do not claim correlation proves causation.
- Recommendations must be clearly presented as
  future investigations.
"""

    response = llm.invoke(
        prompt
    )

    final_report = response.content

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    print()
    print("===== FINAL REPORT GENERATED =====")

    print(final_report)

    # ------------------------------------------------------------
    # RETURN STATE
    # ------------------------------------------------------------

    return {
        "final_report": final_report,

        "model_evidence": model_evidence,

        "evidence": evidence,

        "completed_tools": (
            state.get("completed_tools")
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