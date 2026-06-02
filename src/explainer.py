from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm

_SYSTEM_PROMPT = """\
You are a data analyst communicating insights to a business audience.
Given chart specifications and a dataset profile, write a plain-language explanation for each chart.

For each chart provide exactly three fields:
- what_it_shows: one sentence describing what the chart visualizes
- key_insight: the most important pattern, trend, or anomaly, referencing actual values from the profile
- suggested_action: one concrete action or question the viewer should follow up on

Rules:
- Be specific: reference column names and approximate values from the profile (e.g. "mean salary of 65 000").
- Do NOT invent numbers not present in the profile.
- Keep each field to one sentence.

Respond with ONLY a JSON array — no markdown, no prose. One element per chart, in the same order:
[
  {
    "chart_title": "<exact title from the spec>",
    "what_it_shows": "<one sentence>",
    "key_insight": "<one sentence with specific data references>",
    "suggested_action": "<one actionable sentence>"
  }
]
"""


@dataclass
class ChartExplanation:
    chart_title: str
    what_it_shows: str
    key_insight: str
    suggested_action: str

    def to_dict(self) -> dict:
        return {
            "chart_title": self.chart_title,
            "what_it_shows": self.what_it_shows,
            "key_insight": self.key_insight,
            "suggested_action": self.suggested_action,
        }


def explain_charts(chart_plan: list[dict], profile_text: str) -> list[ChartExplanation]:
    """
    Generate plain-language explanations for each chart in the plan.

    Args:
        chart_plan:   List of ChartSpec dicts (from state["chart_plan"]).
        profile_text: Output of profile_to_text().

    Returns:
        One ChartExplanation per chart, in the same order as chart_plan.
        If the LLM returns fewer items than charts, the remainder are padded
        with empty strings so indices always align.
    """
    if not chart_plan:
        return []

    charts_text = "\n".join(
        f'{i + 1}. [{c["chart_type"].upper()}] "{c["title"]}" '
        f'(x={c.get("x")}, y={c.get("y")}) — {c.get("justification", "")}'
        for i, c in enumerate(chart_plan)
    )

    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Dataset profile:\n\n{profile_text}\n\n"
            f"Charts to explain:\n\n{charts_text}\n\n"
            "Write one explanation per chart, in the same order."
        )),
    ]
    response = llm.invoke(messages)
    return _parse_response(response.content, chart_plan)


def _parse_response(content: str, chart_plan: list[dict]) -> list[ChartExplanation]:
    text = _extract_json(content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw:\n{content}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    explanations: list[ChartExplanation] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        explanations.append(ChartExplanation(
            chart_title=str(item.get("chart_title") or ""),
            what_it_shows=str(item.get("what_it_shows") or ""),
            key_insight=str(item.get("key_insight") or ""),
            suggested_action=str(item.get("suggested_action") or ""),
        ))

    # Pad so indices always align with chart_plan
    while len(explanations) < len(chart_plan):
        explanations.append(ChartExplanation(
            chart_title=chart_plan[len(explanations)].get("title", ""),
            what_it_shows="", key_insight="", suggested_action="",
        ))

    return explanations


def _extract_json(text: str) -> str:
    """Extract a JSON array from LLM output (handles fences and leading prose)."""
    text = text.strip()
    if text.startswith("["):
        return text
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n?```", text)
    if match:
        return match.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text
