import pandas as pd

from datapilot.agent.graph import build_graph
from datapilot.evidence import evaluate_model_comparison
from datapilot.models import run_baseline
from datapilot.tabpfn_tool import run_tabpfn


def main():

    # ============================================================
    # LOAD DATASET
    # ============================================================

    df = pd.read_csv(
        "data/loan_default_sample.csv"
    )

    target = "default"

    print()
    print("=" * 60)
    print("PREPARING MODEL EVIDENCE")
    print("=" * 60)

    # ============================================================
    # RUN BASELINE
    # ============================================================

    baseline = run_baseline(
        df,
        target
    )

    print()
    print("Baseline:")
    print(baseline)

    # ============================================================
    # RUN TABPFN
    # ============================================================

    tabpfn = run_tabpfn(
        df,
        target
    )

    print()
    print("TabPFN:")
    print(tabpfn)

    # ============================================================
    # CALCULATE MODEL EVIDENCE
    # ============================================================

    model_evidence = evaluate_model_comparison(
        [
            baseline,
            tabpfn,
        ]
    )

    print()
    print("Model Evidence:")
    print(model_evidence)

    # ============================================================
    # BUILD LANGGRAPH
    # ============================================================

    graph = build_graph()

    # ============================================================
    # INITIAL STATE
    # ============================================================

    initial_state = {

        "df": df,

        "target": target,

        "problem_type": "classification",

        "profile": {
            "rows": len(df),

            "columns": len(df.columns),

            "numeric_columns": (
                df.select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            ),

            "missing_values": int(
                df.isna()
                .sum()
                .sum()
            ),

            "duplicate_rows": int(
                df.duplicated()
                .sum()
            ),
        },

        # No EDA has been performed by the graph yet.
        "evidence": [],

        # IMPORTANT:
        # Pass deterministic model evidence
        # into the agent state.
        "model_evidence": model_evidence,

        # These models have already been evaluated
        # before starting the graph.
        "completed_tools": [
            "run_baseline",
            "run_tabpfn",
        ],

        "step": 0,

        "max_steps": 5,
    }

    # ============================================================
    # RUN AGENT
    # ============================================================

    result = graph.invoke(
        initial_state
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    print()
    print("=" * 60)
    print("DATAPILOT AGENT RESULT")
    print("=" * 60)

    print()
    print("Next action:")
    print(
        result.get(
            "next_action"
        )
    )

    print()
    print("Hypothesis:")
    print(
        result.get(
            "hypothesis"
        )
    )

    print()
    print("Reasoning:")
    print(
        result.get(
            "reasoning"
        )
    )

    print()
    print("Completed tools:")
    print(
        result.get(
            "completed_tools"
        )
    )

    print()
    print("Model evidence:")
    print(
        result.get(
            "model_evidence"
        )
    )

    print()
    print("Evidence:")
    print(
        result.get(
            "evidence"
        )
    )

    print()
    print("Final report:")
    print(
        result.get(
            "final_report"
        )
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()