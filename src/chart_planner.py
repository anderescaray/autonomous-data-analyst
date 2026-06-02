from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm

VALID_CHART_TYPES = {"bar", "line", "scatter", "pie", "histogram", "heatmap", "boxplot"}

_SYSTEM_PROMPT = """\
You are a data visualization expert. Given a dataset profile, recommend the most insightful charts.

The profile format is:
  Dataset: N rows × M columns
  - column_name (dtype): X% null, Y unique | min=... max=... mean=...   ← numeric
  - column_name (text):  X% null, Y unique | top: val1, val2, val3      ← categorical
  - column_name (datetime): X% null, Y unique | range: YYYY-MM-DD → YYYY-MM-DD
  - column_name (text): ... [high-cardinality — not suitable for charts]

## Rules
- Only use column names that appear in the profile.
- NEVER use a column marked "[high-cardinality — not suitable for charts]".
- datetime column + numeric column → line chart; x = datetime column, y = numeric column.
  If there are multiple numeric columns, recommend one line chart per relevant numeric column.
- one categorical (≤20 unique) + one numeric → bar chart (x = categorical, y = numeric).
- one categorical (≤5 unique) + one numeric that represents a total or share → pie chart.
- numeric distribution → histogram; x = numeric column, y = null.
- one categorical + one numeric with many distinct values → boxplot (shows spread per category).
- two or more numeric columns → scatter (pick the two most informative) or heatmap if ≥4 numeric columns.
- Recommend between 2 and 6 charts. Prefer quality over quantity.
- Every chart must have a short justification grounded in the profile data.

## Output format
Respond with ONLY a JSON array — no markdown, no prose before or after. Each element:
{
  "chart_type": "<bar|line|scatter|pie|histogram|heatmap|boxplot>",
  "title": "<descriptive title>",
  "x": "<column name or null>",
  "y": "<column name or null>",
  "justification": "<one sentence>"
}
"""


@dataclass
class ChartSpec:
    chart_type: str
    title: str
    x: str | None
    y: str | None
    justification: str

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "justification": self.justification,
        }


def plan_charts(profile_text: str) -> list[ChartSpec]:
    """
    Call the configured LLM to decide which charts are relevant for this dataset.

    Args:
        profile_text: Output of profiler.profile_to_text().

    Returns:
        List of ChartSpec objects ordered by recommended priority.
    """
    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Dataset profile:\n\n{profile_text}\n\nRecommend charts."),
    ]
    response = llm.invoke(messages)
    return _parse_response(response.content)


def _parse_response(content: str) -> list[ChartSpec]:
    """Parse and validate the raw LLM JSON output into ChartSpec objects."""
    text = _extract_json(content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw:\n{content}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    specs: list[ChartSpec] = []
    for item in data:
        if not isinstance(item, dict):
            continue  # guard against ["bar", "line"] style responses
        chart_type = str(item.get("chart_type", "")).lower().strip()
        if chart_type not in VALID_CHART_TYPES:
            continue
        specs.append(ChartSpec(
            chart_type=chart_type,
            title=str(item.get("title") or "Chart"),
            x=item.get("x") or None,
            y=item.get("y") or None,
            justification=str(item.get("justification") or ""),
        ))

    return specs


def _extract_json(text: str) -> str:
    """
    Extract a JSON array from raw LLM output.

    Handles three real-world cases:
      1. Clean array with no surrounding text.
      2. Array inside a markdown code fence (anywhere in the text).
      3. Array preceded by prose (model ignores "no prose" instruction).
    """
    text = text.strip()

    # Case 1: already starts with '['
    if text.startswith("["):
        return text

    # Case 2: content inside a code fence, possibly preceded by prose
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n?```", text)
    if match:
        return match.group(1).strip()

    # Case 3: bare array somewhere in the text after prose
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]

    # Nothing found — let json.loads raise with a useful error
    return text
