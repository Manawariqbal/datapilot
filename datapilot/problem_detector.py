import pandas as pd

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = df[target]

    if pd.api.types.is_numeric_dtype(y):
        # Numeric targets with relatively few unique values are often classification.
        if y.nunique(dropna=True) <= 10:
            return "classification"
        return "regression"

    return "classification"
