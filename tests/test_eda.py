import pandas as pd

from datapilot.eda import run_eda


def test_run_eda_numeric_target():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [20, 40, 60, 80],
        "target": [10, 20, 30, 40],
    })
    result = run_eda(df, "target")
    assert result["target_summary"]["type"] == "numeric"
    assert result["correlations_with_target"]
