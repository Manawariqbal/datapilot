import pandas as pd
from datapilot.profiling import profile_dataframe

def test_profile():
    df = pd.DataFrame({"age": [20, 21, None], "city": ["Pune", "Delhi", "Pune"]})
    result = profile_dataframe(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_values"]["age"] == 1
