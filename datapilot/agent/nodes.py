from datapilot.agent.llm import get_llm
from datapilot.eda import run_eda
from datapilot.models import run_baseline
from datapilot.tabpfn_tool import run_tabpfn


ALLOWED_ACTIONS = {
    "run_eda",
    "run_baseline",
    "run_tabpfn",
    "finish",
}


def get_next_missing_action(
    completed_tools: list[str],
) -> str:

    if "run_eda" not in completed_tools:
        return "run_eda"

    if "run_baseline" not in completed_tools:
        return "run_baseline"

    if "run_tabpfn" not in completed_tools:
        return "run_tabpfn"

    return "finish"


def scientist_node(state):

    llm = get_llm()

    completed_tools = (
        state.get("completed_tools")
        or []
    )

    evidence = (
        state.get("evidence")
        or []
    )

    model_evidence = (
        state.get("model_evidence")
        or {}
    )

    evidence_text = "\n".join(
        [
            str(item)
            for item in evidence
        ]
    )

    if not evidence_text:
        evidence_text = (
            "No previous tool evidence available."
        )

    if model_evidence:
        model_evidence_text = str(
            model_evidence
        )
    else:
        model_evidence_text = (
            "No model comparison evidence available yet."
        )

    prompt = f"""
You are DataPilot, an autonomous data scientist.

Your job is to decide what investigation should happen next.

You have access to deterministic Python tools.

============================================================
STRICT SCIENTIFIC RULES
============================================================

1. NEVER invent metrics.

2. NEVER invent capabilities of a tool.

3. Only describe what a tool actually does.

4. Do not claim that TabPFN performs:
   - outlier detection
   - skewness analysis
   - missing-value analysis
   - feature importance
   - statistical testing

   The current TabPFN tool ONLY:
   - trains a TabPFN model
   - makes predictions
   - calculates accuracy
   - calculates weighted F1 for classification
   - calculates MAE and R2 for regression

5. The EDA tool performs:
   - numeric summaries
   - categorical summaries
   - target summary
   - correlations
   - p-values
   - data quality flags

6. The baseline tool trains a baseline ML model.

7. Do not claim causation from correlation.

8. If a tool has already been completed, NEVER select it.

9. If sufficient evidence already exists, choose finish.

10. Use the actual evidence below when making decisions.

============================================================
DATASET
============================================================

Target:
{state.get("target")}

Problem type:
{state.get("problem_type")}

============================================================
COMPLETED TOOLS
============================================================

{completed_tools}

============================================================
PREVIOUS TOOL EVIDENCE
============================================================

{evidence_text}

============================================================
MODEL COMPARISON EVIDENCE
============================================================

{model_evidence_text}

============================================================
AVAILABLE ACTIONS
============================================================

run_eda

Runs exploratory data analysis including:

- descriptive statistics
- categorical summaries
- target statistics
- feature-target correlations
- p-values
- data-quality checks


run_baseline

Runs the baseline machine-learning model and returns
its evaluation metrics.


run_tabpfn

Runs TabPFN and returns model evaluation metrics.

It does NOT perform additional EDA or statistical analysis.


finish

Finish the investigation when the current evidence
is sufficient.

============================================================
DECISION
============================================================

Choose exactly ONE action.

Return EXACTLY:

ACTION: <action>
HYPOTHESIS: <short scientific hypothesis>
REASON: <why this action is the best next step>

Do not include anything else.
"""

    response = llm.invoke(
        prompt
    )

    content = response.content.strip()

    print()
    print("===== GEMMA RESPONSE =====")
    print(content)

    action = "finish"
    hypothesis = ""
    reasoning = ""

    for line in content.splitlines():

        line = line.strip()

        if line.startswith("ACTION:"):
            action = (
                line.split(
                    "ACTION:",
                    1
                )[1]
                .strip()
            )

        elif line.startswith("HYPOTHESIS:"):
            hypothesis = (
                line.split(
                    "HYPOTHESIS:",
                    1
                )[1]
                .strip()
            )

        elif line.startswith("REASON:"):
            reasoning = (
                line.split(
                    "REASON:",
                    1
                )[1]
                .strip()
            )

    if action not in ALLOWED_ACTIONS:

        print(
            f"⚠️ Invalid action from Gemma: {action}"
        )

        action = get_next_missing_action(
            completed_tools
        )

        print(
            f"🔒 Python changed action to: {action}"
        )

    if action in completed_tools:

        print(
            f"⚠️ Gemma selected completed tool: {action}"
        )

        action = get_next_missing_action(
            completed_tools
        )

        print(
            f"🔒 Python changed action to: {action}"
        )

    return {
        "next_action": action,
        "hypothesis": hypothesis,
        "reasoning": reasoning,

        # IMPORTANT:
        # Explicitly preserve state.
        "completed_tools": completed_tools,
        "evidence": evidence,
        "model_evidence": model_evidence,

        "step": state.get(
            "step",
            0
        ),
        "max_steps": state.get(
            "max_steps",
            5
        ),
    }


def eda_node(state):

    print()
    print("===== RUNNING EDA =====")

    df = state["df"]
    target = state["target"]

    result = run_eda(
        df,
        target
    )

    evidence = list(
        state.get("evidence")
        or []
    )

    evidence.append(
        {
            "tool": "run_eda",
            "hypothesis": state.get(
                "hypothesis",
                ""
            ),
            "reasoning": state.get(
                "reasoning",
                ""
            ),
            "result": result,
        }
    )

    completed_tools = list(
        state.get("completed_tools")
        or []
    )

    if "run_eda" not in completed_tools:
        completed_tools.append(
            "run_eda"
        )

    print("EDA completed.")

    return {
        "evidence": evidence,
        "completed_tools": completed_tools,

        # Preserve model evidence.
        "model_evidence": (
            state.get("model_evidence")
            or {}
        ),

        "step": state.get(
            "step",
            0
        ) + 1,
    }


def baseline_node(state):

    print()
    print("===== RUNNING BASELINE =====")

    df = state["df"]
    target = state["target"]

    result = run_baseline(
        df,
        target
    )

    evidence = list(
        state.get("evidence")
        or []
    )

    evidence.append(
        {
            "tool": "run_baseline",
            "hypothesis": state.get(
                "hypothesis",
                ""
            ),
            "reasoning": state.get(
                "reasoning",
                ""
            ),
            "result": result,
        }
    )

    completed_tools = list(
        state.get("completed_tools")
        or []
    )

    if "run_baseline" not in completed_tools:
        completed_tools.append(
            "run_baseline"
        )

    print("Baseline completed.")

    print("Result:")
    print(result)

    return {
        "evidence": evidence,
        "completed_tools": completed_tools,

        "model_evidence": (
            state.get("model_evidence")
            or {}
        ),

        "step": state.get(
            "step",
            0
        ) + 1,
    }


def tabpfn_node(state):

    print()
    print("===== RUNNING TABPFN =====")

    df = state["df"]
    target = state["target"]

    evidence = list(
        state.get("evidence")
        or []
    )

    completed_tools = list(
        state.get("completed_tools")
        or []
    )

    try:

        result = run_tabpfn(
            df,
            target
        )

        print("TabPFN completed.")

        print("Result:")
        print(result)

        evidence.append(
            {
                "tool": "run_tabpfn",
                "hypothesis": state.get(
                    "hypothesis",
                    ""
                ),
                "reasoning": state.get(
                    "reasoning",
                    ""
                ),
                "result": result,
            }
        )

    except Exception as exc:

        result = {
            "error": str(exc)
        }

        print("⚠️ TabPFN failed:")
        print(exc)

        evidence.append(
            {
                "tool": "run_tabpfn",
                "hypothesis": state.get(
                    "hypothesis",
                    ""
                ),
                "reasoning": state.get(
                    "reasoning",
                    ""
                ),
                "result": result,
            }
        )

    if "run_tabpfn" not in completed_tools:
        completed_tools.append(
            "run_tabpfn"
        )

    return {
        "evidence": evidence,
        "completed_tools": completed_tools,

        "model_evidence": (
            state.get("model_evidence")
            or {}
        ),

        "step": state.get(
            "step",
            0
        ) + 1,
    }