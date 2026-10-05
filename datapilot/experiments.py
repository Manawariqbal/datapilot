from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from datapilot.models import run_baseline
from datapilot.eda import run_eda
from datapilot.tabpfn_tool import run_tabpfn
from datapilot.evidence import evaluate_model_comparison


def run_experiments(
    df: pd.DataFrame,
    target: str,
    run_tabpfn_model: bool = True
) -> dict:
    """
    Run DataPilot experiments and evaluate
    the resulting evidence.
    """

    # ----------------------------------------
    # Detect problem type
    # ----------------------------------------

    problem = (
        "classification"
        if (
            not pd.api.types.is_numeric_dtype(
                df[target]
            )
            or df[target].nunique() <= 10
        )
        else "regression"
    )

    results = {
        "run_id": datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ"),

        "problem_type": problem,

        "target": target,

        "eda": run_eda(
            df,
            target
        ),

        "models": [],

        "errors": [],
    }

    # ----------------------------------------
    # Baseline model
    # ----------------------------------------

    baseline = run_baseline(
        df,
        target
    )

    results["models"].append(
        baseline
    )

    # ----------------------------------------
    # TabPFN
    # ----------------------------------------

    if run_tabpfn_model:

        try:

            tabpfn_result = run_tabpfn(
                df,
                target
            )

            results["models"].append(
                tabpfn_result
            )

        except Exception as exc:

            results["errors"].append(
                {
                    "stage": "TabPFN",
                    "error": str(exc),
                }
            )

    # ----------------------------------------
    # Evaluate model evidence
    # ----------------------------------------

    results["model_evidence"] = (
        evaluate_model_comparison(
            results["models"]
        )
    )

    return results


def save_run(
    results: dict,
    base_dir: str = "runs"
) -> Path:

    run_dir = (
        Path(base_dir)
        / f"run_{results['run_id']}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        run_dir / "experiments.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            default=str
        )

    return run_dir