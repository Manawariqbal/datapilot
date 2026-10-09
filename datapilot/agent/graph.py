
from langgraph.graph import END, START, StateGraph

from datapilot.agent.report import generate_final_report

from datapilot.agent.nodes import (
    baseline_node,
    eda_node,
    feature_analysis_node,
    scientist_node,
    tabpfn_node,
)

from datapilot.agent.state import DataPilotState


def route_after_scientist(state):
    """
    Decide which tool the scientist agent should execute next.

    The LLM proposes an action, but Python validates it
    against the tools that have already been completed.
    """

    action = state.get(
        "next_action",
        "finish"
    )

    completed = state.get(
        "completed_tools"
    ) or []

    # ------------------------------------
    # HARD ROUTING GUARD
    # ------------------------------------

    # If Gemma selects a tool that has already
    # been executed, force the next missing tool.
    if action in completed:

        if "run_eda" not in completed:
            return "eda"

        if "run_baseline" not in completed:
            return "baseline"

        if "run_tabpfn" not in completed:
            return "tabpfn"

        if "run_feature_analysis" not in completed:
            return "feature_analysis"

        return "finish"

    # ------------------------------------
    # Normal routing
    # ------------------------------------

    if action == "run_eda":
        return "eda"

    if action == "run_baseline":
        return "baseline"

    if action == "run_tabpfn":
        return "tabpfn"

    if action == "run_feature_analysis":
        return "feature_analysis"

    return "finish"


def route_after_tool(state):
    """
    Decide whether the agent should continue investigating
    or move to the final report.
    """

    step = state.get(
        "step",
        0
    )

    max_steps = state.get(
        "max_steps",
        5
    )

    # Prevent infinite agent loops.
    if step >= max_steps:
        return "finish"

    return "scientist"


def build_graph():
    """
    Build and compile the DataPilot LangGraph workflow.
    """

    graph = StateGraph(
        DataPilotState
    )

    # ------------------------------------
    # Register nodes
    # ------------------------------------

    graph.add_node(
        "scientist",
        scientist_node
    )

    graph.add_node(
        "eda",
        eda_node
    )

    graph.add_node(
        "baseline",
        baseline_node
    )

    graph.add_node(
        "tabpfn",
        tabpfn_node
    )

    graph.add_node(
        "feature_analysis",
        feature_analysis_node
    )

    graph.add_node(
        "report",
        generate_final_report
    )

    # ------------------------------------
    # Entry point
    # ------------------------------------

    graph.add_edge(
        START,
        "scientist"
    )

    # ------------------------------------
    # Scientist routing
    # ------------------------------------

    graph.add_conditional_edges(
        "scientist",
        route_after_scientist,
        {
            "eda": "eda",
            "baseline": "baseline",
            "tabpfn": "tabpfn",
            "feature_analysis": "feature_analysis",
            "finish": "report",
        },
    )

    # ------------------------------------
    # EDA routing
    # ------------------------------------

    graph.add_conditional_edges(
        "eda",
        route_after_tool,
        {
            "scientist": "scientist",
            "finish": "report",
        },
    )

    # ------------------------------------
    # Baseline routing
    # ------------------------------------

    graph.add_conditional_edges(
        "baseline",
        route_after_tool,
        {
            "scientist": "scientist",
            "finish": "report",
        },
    )

    # ------------------------------------
    # TabPFN routing
    # ------------------------------------

    graph.add_conditional_edges(
        "tabpfn",
        route_after_tool,
        {
            "scientist": "scientist",
            "finish": "report",
        },
    )

    # ------------------------------------
    # Feature analysis routing
    # ------------------------------------

    graph.add_conditional_edges(
        "feature_analysis",
        route_after_tool,
        {
            "scientist": "scientist",
            "finish": "report",
        },
    )

    # ------------------------------------
    # Final report
    # ------------------------------------

    graph.add_edge(
        "report",
        END
    )

    return graph.compile()

