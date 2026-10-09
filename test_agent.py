
import pandas as pd

from datapilot.agent.graph import build_graph


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
    print("STARTING DATAPILOT")
    print("=" * 60)

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

        # --------------------------------------------------------
        # BASIC DATASET PROFILE
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # EVIDENCE
        # --------------------------------------------------------

        # Every tool will append its deterministic
        # result to this single evidence list.
        "evidence": [],

        # --------------------------------------------------------
        # MODEL EVIDENCE
        # --------------------------------------------------------

        # Model comparison will be populated later.
        "model_evidence": {},

        # --------------------------------------------------------
        # FEATURE EVIDENCE
        # --------------------------------------------------------

        "feature_evidence": {},

        # --------------------------------------------------------
        # COMPLETED TOOLS
        # --------------------------------------------------------

        # IMPORTANT:
        # Nothing has been executed yet.
        #
        # The agent will decide:
        #
        # run_eda
        # run_baseline
        # run_tabpfn
        # run_feature_analysis
        #
        # and each node will update this list.
        "completed_tools": [],

        # --------------------------------------------------------
        # AGENT LOOP
        # --------------------------------------------------------

        "step": 0,

        "max_steps": 5,
    }

    # ============================================================
    # RUN DATAPILOT
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

    # ------------------------------------------------------------
    # NEXT ACTION
    # ------------------------------------------------------------

    print()
    print("Next action:")
    print(
        result.get(
            "next_action"
        )
    )

    # ------------------------------------------------------------
    # HYPOTHESIS
    # ------------------------------------------------------------

    print()
    print("Hypothesis:")
    print(
        result.get(
            "hypothesis"
        )
    )

    # ------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------

    print()
    print("Reasoning:")
    print(
        result.get(
            "reasoning"
        )
    )

    # ------------------------------------------------------------
    # COMPLETED TOOLS
    # ------------------------------------------------------------

    print()
    print("Completed tools:")
    print(
        result.get(
            "completed_tools"
        )
    )

    # ------------------------------------------------------------
    # MODEL EVIDENCE
    # ------------------------------------------------------------

    print()
    print("Model evidence:")
    print(
        result.get(
            "model_evidence"
        )
    )

    # ------------------------------------------------------------
    # FEATURE EVIDENCE
    # ------------------------------------------------------------

    print()
    print("Feature evidence:")
    print(
        result.get(
            "feature_evidence"
        )
    )

    # ------------------------------------------------------------
    # UNIFIED EVIDENCE
    # ------------------------------------------------------------

    evidence = (
        result.get(
            "evidence"
        )
        or []
    )

    print()
    print("Evidence count:")
    print(
        len(evidence)
    )

    print()
    print("Evidence:")

    for item in evidence:

        print()
        print(
            f"Tool: {item.get('tool')}"
        )

        print(
            "Hypothesis:"
        )

        print(
            item.get(
                "hypothesis"
            )
        )

        print(
            "Reasoning:"
        )

        print(
            item.get(
                "reasoning"
            )
        )

        print(
            "Result:"
        )

        print(
            item.get(
                "result"
            )
        )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print()
    print(
        result.get(
            "final_report"
        )
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
