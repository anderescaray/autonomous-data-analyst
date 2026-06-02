import json
import pytest

from src.chart_planner import ChartSpec, _parse_response, _extract_json, VALID_CHART_TYPES

# ── helpers ───────────────────────────────────────────────────────────────────

def _json(specs: list[dict]) -> str:
    return json.dumps(specs)


VALID_SPEC = {"chart_type": "bar", "title": "Sales by Region", "x": "region", "y": "sales", "justification": "Categorical x, numeric y."}
HISTOGRAM  = {"chart_type": "histogram", "title": "Age Distribution", "x": "age", "y": None, "justification": "Numeric distribution."}
HEATMAP    = {"chart_type": "heatmap", "title": "Correlation Matrix", "x": None, "y": None, "justification": "Many numeric cols."}


# ── _extract_json ─────────────────────────────────────────────────────────────

def test_extract_clean_array():
    raw = '[{"chart_type": "bar"}]'
    assert _extract_json(raw) == raw


def test_extract_json_fence_no_prose():
    raw = "```json\n[]\n```"
    assert _extract_json(raw) == "[]"


def test_extract_plain_fence_no_prose():
    raw = "```\n[]\n```"
    assert _extract_json(raw) == "[]"


def test_extract_json_fence_with_prose_before():
    raw = "Here are my recommendations:\n\n```json\n[]\n```"
    assert _extract_json(raw) == "[]"


def test_extract_prose_then_bare_array():
    raw = 'Sure! Here is the JSON:\n[{"chart_type": "bar"}]'
    result = _extract_json(raw)
    assert result.startswith("[")
    assert "bar" in result


def test_extract_multiline_fence_with_prose():
    inner = '[{"chart_type": "bar", "title": "T", "x": "a", "y": "b", "justification": "j"}]'
    raw = f"Based on the data:\n\n```json\n{inner}\n```\n\nHope this helps!"
    assert _extract_json(raw) == inner


# ── _parse_response — happy paths ─────────────────────────────────────────────

def test_parse_single_valid_spec():
    specs = _parse_response(_json([VALID_SPEC]))
    assert len(specs) == 1
    assert specs[0].chart_type == "bar"
    assert specs[0].x == "region"
    assert specs[0].y == "sales"


def test_parse_histogram_y_is_none():
    specs = _parse_response(_json([HISTOGRAM]))
    assert specs[0].chart_type == "histogram"
    assert specs[0].y is None


def test_parse_heatmap_x_y_none():
    specs = _parse_response(_json([HEATMAP]))
    assert specs[0].x is None
    assert specs[0].y is None


def test_parse_multiple_specs():
    specs = _parse_response(_json([VALID_SPEC, HISTOGRAM, HEATMAP]))
    assert len(specs) == 3


def test_parse_strips_fences_before_parsing():
    raw = f"```json\n{_json([VALID_SPEC])}\n```"
    specs = _parse_response(raw)
    assert len(specs) == 1


def test_parse_handles_prose_before_fence():
    raw = f"Here are my recommendations:\n\n```json\n{_json([VALID_SPEC])}\n```"
    specs = _parse_response(raw)
    assert len(specs) == 1
    assert specs[0].chart_type == "bar"


def test_parse_handles_prose_before_bare_array():
    raw = f"Sure! Here is the JSON:\n{_json([VALID_SPEC])}"
    specs = _parse_response(raw)
    assert len(specs) == 1


def test_parse_skips_non_dict_items():
    raw = '["bar", "line"]'
    specs = _parse_response(raw)
    assert specs == []


def test_parse_all_valid_chart_types_accepted():
    for ct in VALID_CHART_TYPES:
        spec = {**VALID_SPEC, "chart_type": ct}
        result = _parse_response(_json([spec]))
        assert result[0].chart_type == ct


# ── _parse_response — edge cases ─────────────────────────────────────────────

def test_parse_unknown_chart_type_skipped():
    unknown = {**VALID_SPEC, "chart_type": "treemap"}
    specs = _parse_response(_json([unknown, VALID_SPEC]))
    assert len(specs) == 1
    assert specs[0].chart_type == "bar"


def test_parse_empty_array():
    specs = _parse_response("[]")
    assert specs == []


def test_parse_empty_string_y_coerced_to_none():
    spec = {**VALID_SPEC, "y": ""}
    specs = _parse_response(_json([spec]))
    assert specs[0].y is None


# ── _parse_response — error paths ─────────────────────────────────────────────

def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_response("not json at all")


def test_parse_json_object_not_array_raises():
    with pytest.raises(ValueError, match="Expected a JSON array"):
        _parse_response('{"chart_type": "bar"}')


# ── ChartSpec.to_dict ─────────────────────────────────────────────────────────

def test_to_dict_roundtrip():
    spec = ChartSpec(chart_type="bar", title="T", x="col_a", y="col_b", justification="j")
    d = spec.to_dict()
    assert d == {"chart_type": "bar", "title": "T", "x": "col_a", "y": "col_b", "justification": "j"}
