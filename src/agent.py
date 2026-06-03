"""LangGraph agent graph — wires planner, explainer, ReAct agent, and harvester nodes."""
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.state import AgentState
from src.chart_tools import create_chart_tool
from src.llm import get_llm
from src.profiler import profile_dataframe, profile_to_text
from src.chart_planner import plan_charts
from src.explainer import explain_charts

load_dotenv()

_SYSTEM_TEMPLATE = """\
You are an autonomous Senior Data Analyst.
You have access to a pandas DataFrame named 'df'.

## Dataset profile
{profile_text}

## Recommended charts
{chart_plan_text}

## Chart insights (pre-generated)
{chart_explanations_text}

## Instructions
- Answer user queries by writing and executing Python/Pandas code via the 'python_repl' tool.
- Always execute code to find answers — never guess.
- Use print() to output text results.
- If you encounter an error, diagnose it, fix the code, and retry.
- When generating a chart, use the exact column names from the profile above.
- Call plt.show() after every matplotlib chart so it is captured.
- When presenting a chart to the user, include its pre-generated insight and suggested action from the section above.
- Explain results clearly after executing code.
"""


def _fmt_chart_explanations(explanations: list[dict]) -> str:
    """Render chart explanations as a formatted string for the agent system prompt."""
    if not explanations:
        return "No chart insights available yet."
    lines = []
    for exp in explanations:
        title = exp.get("chart_title", "Chart")
        lines.append(
            f"**{title}**\n"
            f"  - Shows: {exp.get('what_it_shows', '')}\n"
            f"  - Key insight: {exp.get('key_insight', '')}\n"
            f"  - Action: {exp.get('suggested_action', '')}"
        )
    return "\n\n".join(lines)


def _fmt_chart_plan(chart_plan: list[dict]) -> str:
    """Render the chart plan as a numbered list for the agent system prompt."""
    if not chart_plan:
        return "No chart plan available yet."
    lines = []
    for i, spec in enumerate(chart_plan, 1):
        y_part = f" vs {spec['y']}" if spec.get("y") else ""
        lines.append(
            f"{i}. [{spec['chart_type'].upper()}] {spec['title']} "
            f"(x={spec.get('x')}{y_part}) — {spec['justification']}"
        )
    return "\n".join(lines)


def _make_harvester_node(captured: list[dict]):
    """Return a LangGraph node that drains captured figures into state after each tool call."""
    def harvester_node(state: AgentState) -> dict:
        new_figures = captured[:]
        del captured[:]
        if new_figures:
            return {"captured_figures": new_figures}
        return {}
    return harvester_node


def create_agent_graph(df: pd.DataFrame):
    """
    Build and compile the LangGraph agent.

    Returns:
        app:    Compiled LangGraph application.
        p_text: Profile text for the DataFrame (for UI display and initial state).

    Figures produced during tool calls accumulate in state["captured_figures"].
    """
    profile  = profile_dataframe(df)
    p_text   = profile_to_text(profile)

    tool, captured = create_chart_tool(df)
    tools           = [tool]
    llm             = get_llm(temperature=0)
    llm_with_tools  = llm.bind_tools(tools)

    # ── nodes ─────────────────────────────────────────────────────────────────

    def planner_node(state: AgentState) -> dict:
        if state.get("chart_plan"):
            return {}
        specs = plan_charts(state["profile_text"])
        return {"chart_plan": [s.to_dict() for s in specs]}

    def explainer_node(state: AgentState) -> dict:
        if state.get("chart_explanations"):
            return {}
        explanations = explain_charts(state["chart_plan"], state["profile_text"])
        return {"chart_explanations": [e.to_dict() for e in explanations]}

    def agent_node(state: AgentState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain  = prompt | llm_with_tools
        result = chain.invoke({
            "messages":            state["messages"],
            "profile_text":        state.get("profile_text", ""),
            "chart_plan_text":     _fmt_chart_plan(state.get("chart_plan", [])),
            "chart_explanations_text": _fmt_chart_explanations(state.get("chart_explanations", [])),
        })
        return {"messages": [result]}

    tool_node      = ToolNode(tools)
    harvester_node = _make_harvester_node(captured)

    # ── graph ─────────────────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)
    workflow.add_node("planner",   planner_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("agent",     agent_node)
    workflow.add_node("tools",     tool_node)
    workflow.add_node("harvester", harvester_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner",   "explainer")
    workflow.add_edge("explainer", "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools",     "harvester")
    workflow.add_edge("harvester", "agent")

    app = workflow.compile()
    return app, p_text
