from typing import Any


def evaluate_model_comparison(
    models: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compare model results and produce structured evidence.

    This function performs deterministic evaluation.
    The LLM does not calculate the metrics.
    """

    if len(models) < 2:
        return {
            "status": "insufficient_models",
            "message": (
                "At least two models are required "
                "for comparison."
            ),
        }

    baseline = models[0]
    challenger = models[1]

    result = {
        "status": "completed",
        "baseline_model": baseline.get("model"),
        "challenger_model": challenger.get("model"),
    }

    # ---------------------------------------------------------
    # Classification comparison
    # ---------------------------------------------------------

    if (
        "accuracy" in baseline
        and "accuracy" in challenger
    ):
        baseline_accuracy = baseline["accuracy"]
        challenger_accuracy = challenger["accuracy"]

        baseline_f1 = baseline.get("f1_weighted")
        challenger_f1 = challenger.get("f1_weighted")

        accuracy_difference = (
            challenger_accuracy
            - baseline_accuracy
        )

        result["baseline_accuracy"] = baseline_accuracy
        result["challenger_accuracy"] = challenger_accuracy

        result["accuracy_difference"] = (
            accuracy_difference
        )

        if accuracy_difference > 0:
            result["accuracy_winner"] = (
                challenger["model"]
            )

        elif accuracy_difference < 0:
            result["accuracy_winner"] = (
                baseline["model"]
            )

        else:
            result["accuracy_winner"] = "tie"

        # -----------------------------------------------------
        # F1 comparison
        # -----------------------------------------------------

        if (
            baseline_f1 is not None
            and challenger_f1 is not None
        ):
            f1_difference = (
                challenger_f1
                - baseline_f1
            )

            result["baseline_f1_weighted"] = (
                baseline_f1
            )

            result["challenger_f1_weighted"] = (
                challenger_f1
            )

            result["f1_difference"] = (
                f1_difference
            )

            if f1_difference > 0:
                result["f1_winner"] = (
                    challenger["model"]
                )

            elif f1_difference < 0:
                result["f1_winner"] = (
                    baseline["model"]
                )

            else:
                result["f1_winner"] = "tie"

        # -----------------------------------------------------
        # Overall conclusion
        # -----------------------------------------------------

        accuracy_close = (
            abs(accuracy_difference) < 0.01
        )

        f1_close = (
            baseline_f1 is None
            or abs(f1_difference) < 0.01
        )

        if accuracy_close and f1_close:
            result["conclusion"] = (
                "No clear overall improvement. "
                "The two models have very similar "
                "performance."
            )

        elif accuracy_difference > 0:
            result["conclusion"] = (
                f"{challenger['model']} has "
                "higher accuracy."
            )

        else:
            result["conclusion"] = (
                f"{baseline['model']} has "
                "higher accuracy."
            )

    # ---------------------------------------------------------
    # Regression comparison
    # ---------------------------------------------------------

    elif (
        "r2" in baseline
        and "r2" in challenger
    ):
        baseline_r2 = baseline["r2"]
        challenger_r2 = challenger["r2"]

        baseline_mae = baseline.get("mae")
        challenger_mae = challenger.get("mae")

        r2_difference = (
            challenger_r2
            - baseline_r2
        )

        result["baseline_r2"] = baseline_r2
        result["challenger_r2"] = challenger_r2

        result["r2_difference"] = (
            r2_difference
        )

        if r2_difference > 0:
            result["r2_winner"] = (
                challenger["model"]
            )

        elif r2_difference < 0:
            result["r2_winner"] = (
                baseline["model"]
            )

        else:
            result["r2_winner"] = "tie"

        if (
            baseline_mae is not None
            and challenger_mae is not None
        ):
            mae_difference = (
                challenger_mae
                - baseline_mae
            )

            result["baseline_mae"] = (
                baseline_mae
            )

            result["challenger_mae"] = (
                challenger_mae
            )

            result["mae_difference"] = (
                mae_difference
            )

            if mae_difference < 0:
                result["mae_winner"] = (
                    challenger["model"]
                )

            elif mae_difference > 0:
                result["mae_winner"] = (
                    baseline["model"]
                )

            else:
                result["mae_winner"] = "tie"

        if abs(r2_difference) < 0.01:
            result["conclusion"] = (
                "No clear overall improvement. "
                "The two models have very similar "
                "R² performance."
            )

        elif r2_difference > 0:
            result["conclusion"] = (
                f"{challenger['model']} has "
                "higher R²."
            )

        else:
            result["conclusion"] = (
                f"{baseline['model']} has "
                "higher R²."
            )

    else:
        result["status"] = "unsupported_comparison"
        result["message"] = (
            "The provided model results do not "
            "contain compatible comparison metrics."
        )

    return result