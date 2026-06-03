"""Shared state definition for the LangGraph agent graph."""
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Mutable state threaded through every node of the agent graph."""

    messages:            Annotated[list[BaseMessage], add_messages]
    profile_text:        str         # output of profile_to_text(); injected at graph start
    chart_plan:          list[dict]  # ChartSpec dicts; set by planner_node (once)
    chart_explanations:  list[dict]  # ChartExplanation dicts; set by explainer_node (once)
    captured_figures:    Annotated[list[dict], operator.add]  # accumulated by VI-02
