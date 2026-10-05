from typing import Any, TypedDict


class DataPilotState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    # ----------------------------------------
    # Dataset
    # ----------------------------------------

    df: Any

    target: str

    profile: dict

    problem_type: str

    # ----------------------------------------
    # Agent reasoning
    # ----------------------------------------

    hypothesis: str

    next_action: str

    reasoning: str

    # ----------------------------------------
    # Scientific evidence
    # ----------------------------------------

    evidence: list[dict]

    model_evidence: dict

    # ----------------------------------------
    # Execution tracking
    # ----------------------------------------

    completed_tools: list[str]

    step: int

    max_steps: int

    # ----------------------------------------
    # Final output
    # ----------------------------------------

    final_report: str