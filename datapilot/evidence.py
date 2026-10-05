from typing import Any


def evaluate_model_comparison(
    models: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compare model results and produce structured evidence.

    This function performs deterministic evaluation.
    The LLM does not calculate the metrics.
    """

    # ----------------------------------------
    # Validate number of models
    # ----------------------------------------

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

    # ----------------------------------------
    # Classification comparison
    # ----------------------------------------

    if (
        "accuracy" in baseline
        and "accuracy" in challenger
    ):

        baseline_accuracy = baseline["accuracy"]
        challenger_accuracy = challenger["accuracy"]

        baseline_f1 = baseline.get(
            "f1_weighted"
        )

        challenger_f1 = challenger.get(
            "f1_weighted"
        )

        # ------------------------------------
        # Accuracy difference
        # ------------------------------------

        accuracy_difference = (
            challenger_accuracy
            - baseline_accuracy
        )

        result["accuracy_difference"] = (
            accuracy_difference
        )

        # ------------------------------------
        # Accuracy winner
        # ------------------------------------

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

        # ------------------------------------
        # F1 comparison
        # ------------------------------------

        if (
            baseline_f1 is not None
            and challenger_f1 is not None
        ):

            f1_difference = (
                challenger_f1
                - baseline_f1
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

        # ------------------------------------
        # Overall conclusion
        # ------------------------------------

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

    return result