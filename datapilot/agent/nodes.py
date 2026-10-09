
from datapilot.agent.llm import get_llm
from datapilot.eda import run_eda
from datapilot.models import run_baseline
from datapilot.tabpfn_tool import run_tabpfn
from datapilot.feature_analysis import analyze_numeric_features
from datapilot.evidence import evaluate_model_comparison


ALLOWED_ACTIONS = {
    "run_eda",
    "run_baseline",
    "run_tabpfn",
    "run_feature_analysis",
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

    if "run_feature_analysis" not in completed_tools:
        return "run_feature_analysis"

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

    feature_evidence = (
        state.get("feature_evidence")
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

    if feature_evidence:
        feature_evidence_text = str(
            feature_evidence
        )
    else:
        feature_evidence_text = (
            "No feature analysis evidence available yet."
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
   - feature-target correlations
   - p-values
   - data quality flags

6. The baseline tool trains a baseline ML model.

7. The feature analysis tool performs:
   - Pearson correlation
   - p-value calculation
   - direction detection
   - statistical significance classification
   - sample size reporting

8. Do not claim causation from correlation.

9. If a tool has already been completed, NEVER select it.

10. If sufficient evidence already exists, choose finish.

11. Use the actual evidence below when making decisions.

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
FEATURE ANALYSIS EVIDENCE
============================================================

{feature_evidence_text}

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


run_feature_analysis

Runs deterministic Pearson correlation analysis between
numeric features and the numeric target.

Returns:

- correlation
- p-value
- direction
- statistical significance
- sample size

It does NOT establish causation.


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

    # Enforce the baseline-before-TabPFN dependency so that
    # model comparison always has a baseline result available.
    if (
        action == "run_tabpfn"
        and "run_baseline" not in completed_tools
    ):
        action = "run_baseline"

        print(
            "🔒 Python enforced prerequisite: "
            "run_baseline must run before run_tabpfn."
        )

    return {
        "next_action": action,
        "hypothesis": hypothesis,
        "reasoning": reasoning,

        # Explicitly preserve state.
        "completed_tools": completed_tools,
        "evidence": evidence,
        "model_evidence": model_evidence,
        "feature_evidence": feature_evidence,

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

        # Preserve feature evidence.
        "feature_evidence": (
            state.get("feature_evidence")
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

        "feature_evidence": (
            state.get("feature_evidence")
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
        state.get("evidence") or []
    )

    completed_tools = list(
        state.get("completed_tools") or []
    )

    model_evidence = (
        state.get("model_evidence") or {}
    )

    feature_evidence = (
        state.get("feature_evidence") or {}
    )

    # ------------------------------------------------------------
    # RUN TABPFN
    # ------------------------------------------------------------

    try:
        result = run_tabpfn(
            df,
            target
        )

        print("TabPFN completed.")
        print("Result:")
        print(result)

    except Exception as exc:
        result = {
            "status": "failed",
            "model": "TabPFN",
            "problem_type": state.get("problem_type"),
            "error": str(exc),
        }

        print("⚠️ TabPFN failed:")
        print(exc)

    # ------------------------------------------------------------
    # APPEND TABPFN RESULT TO UNIFIED EVIDENCE
    # ------------------------------------------------------------

    evidence.append({
        "tool": "run_tabpfn",
        "hypothesis": state.get("hypothesis", ""),
        "reasoning": state.get("reasoning", ""),
        "result": result,
    })

    if "run_tabpfn" not in completed_tools:
        completed_tools.append("run_tabpfn")

    # ------------------------------------------------------------
    # FIND THE BASELINE RESULT
    # ------------------------------------------------------------

    baseline_result = None

    for item in evidence:
        if item.get("tool") == "run_baseline":
            candidate = item.get("result") or {}

            if (
                "error" not in candidate
                and candidate.get("status") != "failed"
            ):
                baseline_result = candidate

            break

    # ------------------------------------------------------------
    # CALCULATE DETERMINISTIC MODEL COMPARISON
    # ------------------------------------------------------------

    tabpfn_succeeded = (
        "error" not in result
        and result.get("status") != "failed"
        and ("accuracy" in result or "r2" in result)
    )

    if baseline_result is not None and tabpfn_succeeded:
        model_evidence = evaluate_model_comparison([
            baseline_result,
            result,
        ])

        print()
        print("===== MODEL COMPARISON =====")
        print(model_evidence)

        evidence.append({
            "tool": "evaluate_model_comparison",
            "hypothesis": (
                "Compare baseline and TabPFN performance "
                "using deterministic evaluation metrics."
            ),
            "reasoning": (
                "Both model results are available. Python calculates "
                "metric differences and identifies the metric winners."
            ),
            "result": model_evidence,
        })
    else:
        print(
            "Model comparison skipped: successful baseline and "
            "TabPFN results with compatible metrics are required."
        )

    # ------------------------------------------------------------
    # RETURN UPDATED STATE
    # ------------------------------------------------------------

    return {
        "evidence": evidence,
        "completed_tools": completed_tools,
        "model_evidence": model_evidence,
        "feature_evidence": feature_evidence,
        "step": state.get("step", 0) + 1,
    }


def feature_analysis_node(state):

    print()
    print("===== RUNNING FEATURE ANALYSIS =====")

    df = state["df"]
    target = state["target"]

    result = analyze_numeric_features(
        df,
        target,
    )

    evidence = list(
        state.get("evidence")
        or []
    )

    evidence.append(
        {
            "tool": "run_feature_analysis",
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

    if "run_feature_analysis" not in completed_tools:
        completed_tools.append(
            "run_feature_analysis"
        )

    print("Feature analysis completed.")

    print("Result:")
    print(result)

    return {
        "evidence": evidence,
        "completed_tools": completed_tools,

        # Store the latest structured feature analysis.
        "feature_evidence": result,

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
