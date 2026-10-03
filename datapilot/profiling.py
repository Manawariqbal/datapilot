import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": categorical.columns.tolist(),
        "missing_values": {
            k: int(v) for k, v in missing[missing > 0].items()
        },
        "duplicate_rows": duplicates,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }
