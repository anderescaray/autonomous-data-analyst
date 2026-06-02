import json
import pytest

from src.explainer import ChartExplanation, _parse_response, _extract_json

PLAN = [
    {"chart_type": "bar",  "title": "Sales by Region",    "x": "region", "y": "sales",    "justification": "j1"},
    {"chart_type": "line", "title": "Revenue Over Time",  "x": "date",   "y": "revenue",  "justification": "j2"},
]

VALID_RESPONSE = json.dumps([
    {"chart_title": "Sales by Region",   "what_it_shows": "A.", "key_insight": "B.", "suggested_action": "C."},
    {"chart_title": "Revenue Over Time", "what_it_shows": "D.", "key_insight": "E.", "suggested_action": "F."},
])


# ── _extract_json (same robustness as chart_planner) ─────────────────────────

def test_extract_clean_array():
    assert _extract_json("[]") == "[]"

def test_extract_fence():
    raw = "```json\n[]\n```"
    assert _extract_json(raw) == "[]"

def test_extract_prose_before_fence():
    raw = "Here you go:\n\n```json\n[]\n```"
    assert _extract_json(raw) == "[]"

def test_extract_prose_before_bare_array():
    raw = f"Sure!\n{VALID_RESPONSE}"
    result = _extract_json(raw)
    assert result.startswith("[")


# ── _parse_response — happy paths ─────────────────────────────────────────────

def test_parse_returns_correct_count():
    result = _parse_response(VALID_RESPONSE, PLAN)
    assert len(result) == 2


def test_parse_fields_mapped():
    result = _parse_response(VALID_RESPONSE, PLAN)
    assert result[0].chart_title == "Sales by Region"
    assert result[0].what_it_shows == "A."
    assert result[0].key_insight == "B."
    assert result[0].suggested_action == "C."


def test_parse_second_item():
    result = _parse_response(VALID_RESPONSE, PLAN)
    assert result[1].chart_title == "Revenue Over Time"


# ── padding when LLM returns fewer items ─────────────────────────────────────

def test_parse_pads_missing_explanations():
    one_item = json.dumps([
        {"chart_title": "Sales by Region", "what_it_shows": "A.", "key_insight": "B.", "suggested_action": "C."},
    ])
    result = _parse_response(one_item, PLAN)
    assert len(result) == 2
    assert result[1].chart_title == "Revenue Over Time"
    assert result[1].what_it_shows == ""


def test_parse_empty_plan_returns_empty():
    result = _parse_response("[]", [])
    assert result == []


# ── error paths ───────────────────────────────────────────────────────────────

def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_response("not json", PLAN)


def test_parse_object_not_array_raises():
    with pytest.raises(ValueError, match="Expected a JSON array"):
        _parse_response('{"chart_title": "x"}', PLAN)


def test_parse_skips_non_dict_items():
    mixed = json.dumps(["string_item", {"chart_title": "Sales by Region", "what_it_shows": "A.", "key_insight": "B.", "suggested_action": "C."}])
    result = _parse_response(mixed, PLAN[:1])
    assert len(result) == 1
    assert result[0].chart_title == "Sales by Region"


# ── ChartExplanation.to_dict ──────────────────────────────────────────────────

def test_to_dict_roundtrip():
    exp = ChartExplanation(chart_title="T", what_it_shows="A", key_insight="B", suggested_action="C")
    assert exp.to_dict() == {"chart_title": "T", "what_it_shows": "A", "key_insight": "B", "suggested_action": "C"}
