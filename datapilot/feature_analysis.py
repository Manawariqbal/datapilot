from typing import Any

import pandas as pd
from scipy.stats import pearsonr


def analyze_numeric_features(
    df: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    """
    Analyze relationships between numeric features
    and a binary/numeric target.

    The analysis is deterministic and does not use an LLM.

    For numeric features:
    - Pearson correlation
    - p-value
    - sample size

    For binary numeric targets, this helps identify
    features that have statistical association with
    the target.

    Important:
    Correlation does NOT imply causation.
    """

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column != target
    ]

    if not feature_columns:
        return {
            "status": "no_numeric_features",
            "target": target,
            "features": [],
        }

    target_series = df[target]

    if not pd.api.types.is_numeric_dtype(
        target_series
    ):
        return {
            "status": "unsupported_target",
            "target": target,
            "message": (
                "Numeric feature analysis currently "
                "requires a numeric target."
            ),
            "features": [],
        }

    results = []

    for feature in feature_columns:

        data = df[
            [feature, target]
        ].dropna()

        if len(data) < 3:
            continue

        x = data[feature]
        y = data[target]

        # Pearson correlation requires variation
        # in both variables.
        if x.nunique() < 2:
            continue

        if y.nunique() < 2:
            continue

        try:
            correlation, p_value = pearsonr(
                x,
                y,
            )
        except Exception:
            continue

        results.append(
            {
                "feature": feature,
                "correlation": float(
                    correlation
                ),
                "p_value": float(
                    p_value
                ),
                "samples": int(
                    len(data)
                ),
            }
        )

    # Strongest absolute correlations first.
    results.sort(
        key=lambda item: abs(
            item["correlation"]
        ),
        reverse=True,
    )

    for item in results:

        correlation = item[
            "correlation"
        ]

        p_value = item[
            "p_value"
        ]

        if p_value < 0.05:
            significance = (
                "statistically_significant"
            )
        else:
            significance = (
                "not_statistically_significant"
            )

        if correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "none"

        item["direction"] = direction
        item["significance"] = significance

    return {
        "status": "completed",
        "target": target,
        "method": "pearson_correlation",
        "features": results,
        "interpretation_note": (
            "Statistical association does not "
            "imply causation."
        ),
    }