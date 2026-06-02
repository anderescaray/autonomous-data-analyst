from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool


def create_chart_tool(df: pd.DataFrame) -> tuple[PythonAstREPLTool, list[dict]]:
    """
    Create a Python REPL tool with matplotlib and (optionally) plotly available.

    plt.show() is monkey-patched so any chart the agent generates is automatically
    captured as PNG bytes.  For plotly figures the agent calls save_figure(fig).

    Returns:
        tool:     LangChain tool to bind to the LLM.
        captured: Shared list that accumulates figure dicts
                  {"format": "png", "data": bytes} or {"format": "html", "data": str}.
                  Callers read this list after tool invocations to retrieve figures.
    """
    captured: list[dict] = []

    def _capture_matplotlib() -> None:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        captured.append({"format": "png", "data": buf.getvalue()})
        plt.close("all")

    def _capture_plotly(fig) -> None:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        captured.append({"format": "html", "data": html})

    plt.show = _capture_matplotlib

    local_vars: dict = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "matplotlib": matplotlib,
        "save_figure": _capture_plotly,
    }

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        local_vars["px"] = px
        local_vars["go"] = go
    except ImportError:
        pass

    tool = PythonAstREPLTool(locals=local_vars)
    tool.name = "python_repl"
    tool.description = (
        "A Python shell for data analysis and chart generation. "
        "Available globals: df (pandas DataFrame), plt (matplotlib.pyplot), "
        "px (plotly.express), go (plotly.graph_objects). "
        "To generate a chart with matplotlib: write the plot code and call plt.show() — "
        "the figure is captured automatically. "
        "For plotly: build a Figure object and call save_figure(fig). "
        "Always use print() to output non-chart text results."
    )
    return tool, captured
