from typing import Any, TypedDict

class DataPilotState(TypedDict, total=False):
    dataset_path: str
    dataframe: Any
    profile: dict
    problem_type: str
    target_column: str
    data_quality: dict
    experiments: list[dict]
    model_results: list[dict]
    insights: list[str]
    report: str
