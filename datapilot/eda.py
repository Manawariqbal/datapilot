import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def run_eda(df: pd.DataFrame, target: str) -> dict:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()

    result = {
        "numeric_summary": {},
        "categorical_summary": {},
        "target_summary": {},
        "correlations_with_target": [],
        "quality_flags": [],
    }

    for col in numeric:
        s = df[col].dropna()
        if len(s):
            result["numeric_summary"][col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
            }

    for col in categorical:
        s = df[col]
        counts = s.value_counts(dropna=False).head(10)
        result["categorical_summary"][col] = {
            "unique": int(s.nunique(dropna=True)),
            "missing": int(s.isna().sum()),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }

    if target in numeric:
        y = pd.to_numeric(df[target], errors="coerce")
        result["target_summary"] = {
            "type": "numeric",
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std()) if y.notna().sum() > 1 else 0.0,
        }

        for col in numeric:
            if col == target:
                continue
            pair = df[[col, target]].dropna()
            if len(pair) >= 3 and pair[col].nunique() > 1:
                r, p = pearsonr(pair[col], pair[target])
                result["correlations_with_target"].append({
                    "feature": col,
                    "method": "pearson",
                    "correlation": float(r),
                    "p_value": float(p),
                    "n": int(len(pair)),
                })
    else:
        counts = df[target].value_counts(dropna=False)
        result["target_summary"] = {
            "type": "categorical",
            "classes": int(df[target].nunique(dropna=True)),
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
        }

        # Numeric feature ↔ binary target: compare class means.
        if df[target].nunique(dropna=True) == 2:
            classes = list(df[target].dropna().unique())
            for col in numeric:
                if col == target:
                    continue
                pair = df[[col, target]].dropna()
                if len(pair) < 4 or pair[col].nunique() <= 1:
                    continue
                means = pair.groupby(target)[col].mean()
                result["correlations_with_target"].append({
                    "feature": col,
                    "method": "class_mean_difference",
                    "class_0": str(classes[0]),
                    "class_1": str(classes[1]),
                    "mean_0": float(means.get(classes[0], np.nan)),
                    "mean_1": float(means.get(classes[1], np.nan)),
                    "absolute_difference": float(
                        abs(means.get(classes[0], np.nan) - means.get(classes[1], np.nan))
                    ),
                })

    # Simple, explainable quality flags.
    if df.duplicated().sum():
        result["quality_flags"].append(
            f"{int(df.duplicated().sum())} duplicate rows detected."
        )

    for col in df.columns:
        missing_pct = float(df[col].isna().mean() * 100)
        if missing_pct >= 20:
            result["quality_flags"].append(
                f"{col} has {missing_pct:.1f}% missing values."
            )
        if df[col].nunique(dropna=True) <= 1:
            result["quality_flags"].append(
                f"{col} has one or fewer unique non-null values."
            )

    result["correlations_with_target"] = sorted(
        result["correlations_with_target"],
        key=lambda x: abs(
            x.get("correlation", x.get("absolute_difference", 0))
        ),
        reverse=True,
    )
    return result
