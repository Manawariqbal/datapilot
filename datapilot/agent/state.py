from typing import Any, TypedDict


class DataPilotState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    df: Any

    target: str

    profile: dict

    problem_type: str

    hypothesis: str

    next_action: str

    reasoning: str

    evidence: list[dict]

    # Deterministic comparison between models.
    model_evidence: dict

    # Deterministic statistical feature analysis.
    feature_evidence: dict

    completed_tools: list[str]

    step: int

    max_steps: int

    final_report: str