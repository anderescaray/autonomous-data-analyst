import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.chart_tools import create_chart_tool

SAMPLE = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_returns_tool_and_captured_list():
    tool, captured = create_chart_tool(SAMPLE)
    assert tool is not None
    assert isinstance(captured, list)


def test_df_in_tool_locals():
    tool, _ = create_chart_tool(SAMPLE)
    result = tool.run("print(len(df))")
    assert "3" in result


def test_matplotlib_in_tool_locals():
    tool, _ = create_chart_tool(SAMPLE)
    result = tool.run("print(type(plt).__name__)")
    assert "module" in result


def test_plt_show_captures_png():
    tool, captured = create_chart_tool(SAMPLE)
    tool.run("plt.plot(df['x'], df['y']); plt.show()")
    assert len(captured) == 1
    assert captured[0]["format"] == "png"
    assert isinstance(captured[0]["data"], bytes)
    assert len(captured[0]["data"]) > 0


def test_multiple_shows_accumulate():
    tool, captured = create_chart_tool(SAMPLE)
    tool.run("plt.plot([1,2]); plt.show()")
    tool.run("plt.bar([1,2],[3,4]); plt.show()")
    assert len(captured) == 2


def test_captured_png_is_valid_image():
    tool, captured = create_chart_tool(SAMPLE)
    tool.run("plt.plot(df['x']); plt.show()")
    data = captured[0]["data"]
    # PNG files start with the PNG magic bytes
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_plotly_in_locals_if_available():
    try:
        import plotly.express as px
    except ImportError:
        return  # plotly not installed, skip
    tool, _ = create_chart_tool(SAMPLE)
    result = tool.run("print(type(px).__name__)")
    assert "module" in result


def test_save_figure_captures_plotly_html():
    try:
        import plotly.express as px
    except ImportError:
        return
    tool, captured = create_chart_tool(SAMPLE)
    tool.run("fig = px.line(df, x='x', y='y'); save_figure(fig)")
    assert len(captured) == 1
    assert captured[0]["format"] == "html"
    assert "<div" in captured[0]["data"]


def test_save_figure_propagates_error_on_invalid_object():
    tool, captured = create_chart_tool(SAMPLE)
    result = tool.run("save_figure('not_a_figure')")
    assert len(captured) == 0
    assert "Error" in result or "error" in result or "AttributeError" in result
