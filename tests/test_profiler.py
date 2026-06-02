import pandas as pd
import pytest

from src.profiler import profile_dataframe, profile_to_text


MIXED = pd.DataFrame({
    "age":    [25, 30, None, 40],
    "salary": [50000.0, 60000.0, 70000.0, 80000.0],
    "city":   ["Madrid", "Madrid", "Barcelona", "Sevilla"],
    "active": [True, False, True, True],
})

DATES = pd.DataFrame({
    "date":  pd.to_datetime(["2020-01-01", "2021-06-15", "2022-12-31"]),
    "value": [100, 200, 300],
})

HIGH_CARD = pd.DataFrame({
    "id":    [f"user_{i}" for i in range(100)],
    "score": range(100),
})


# ── profile_dataframe ─────────────────────────────────────────────────────────

def test_shape():
    p = profile_dataframe(MIXED)
    assert p["shape"] == {"rows": 4, "cols": 4}


def test_numeric_column_has_stats():
    p = profile_dataframe(MIXED)
    salary = next(c for c in p["columns"] if c["name"] == "salary")
    assert salary["dtype"] == "float"
    assert "stats" in salary
    assert "top_values" not in salary
    assert salary["stats"]["min"] == "50000.0"
    assert salary["stats"]["max"] == "80000.0"


def test_text_column_has_top_values():
    p = profile_dataframe(MIXED)
    city = next(c for c in p["columns"] if c["name"] == "city")
    assert city["dtype"] == "text"
    assert "top_values" in city
    assert "stats" not in city
    assert "Madrid" in city["top_values"]


def test_null_percentage():
    p = profile_dataframe(MIXED)
    age = next(c for c in p["columns"] if c["name"] == "age")
    assert age["null_pct"] == 25.0


def test_cardinality():
    p = profile_dataframe(MIXED)
    city = next(c for c in p["columns"] if c["name"] == "city")
    assert city["cardinality"] == 3


def test_sample_excludes_nulls():
    p = profile_dataframe(MIXED)
    age = next(c for c in p["columns"] if c["name"] == "age")
    assert None not in age["sample"]


def test_boolean_dtype():
    p = profile_dataframe(MIXED)
    active = next(c for c in p["columns"] if c["name"] == "active")
    assert active["dtype"] == "boolean"


def test_all_columns_present():
    p = profile_dataframe(MIXED)
    names = [c["name"] for c in p["columns"]]
    assert names == list(MIXED.columns)


# ── profile_to_text ───────────────────────────────────────────────────────────

def test_text_contains_shape():
    text = profile_to_text(profile_dataframe(MIXED))
    assert "4 rows" in text
    assert "4 columns" in text


def test_text_contains_column_names():
    text = profile_to_text(profile_dataframe(MIXED))
    for col in MIXED.columns:
        assert col in text


def test_text_contains_stats_for_numeric():
    text = profile_to_text(profile_dataframe(MIXED))
    assert "min=" in text
    assert "max=" in text
    assert "mean=" in text


def test_text_is_string():
    result = profile_to_text(profile_dataframe(MIXED))
    assert isinstance(result, str)
    assert len(result) > 0


def test_text_no_double_newline_at_header():
    text = profile_to_text(profile_dataframe(MIXED))
    assert not text.startswith("Dataset:\n\n") and "\n\n" not in text[:50]


# ── datetime ──────────────────────────────────────────────────────────────────

def test_datetime_column_has_range():
    p = profile_dataframe(DATES)
    col = next(c for c in p["columns"] if c["name"] == "date")
    assert col["dtype"] == "datetime"
    assert "range" in col
    assert "top_values" not in col


def test_datetime_range_values():
    p = profile_dataframe(DATES)
    col = next(c for c in p["columns"] if c["name"] == "date")
    assert col["range"]["min"] == "2020-01-01"
    assert col["range"]["max"] == "2022-12-31"


def test_text_datetime_shows_range():
    text = profile_to_text(profile_dataframe(DATES))
    assert "range:" in text
    assert "2020-01-01" in text
    assert "2022-12-31" in text


def test_text_datetime_has_no_stats_format():
    text = profile_to_text(profile_dataframe(DATES))
    date_line = next(l for l in text.splitlines() if "date" in l and "datetime" in l)
    assert "min=" not in date_line


# ── high cardinality ──────────────────────────────────────────────────────────

def test_high_cardinality_flag():
    p = profile_dataframe(HIGH_CARD)
    col = next(c for c in p["columns"] if c["name"] == "id")
    assert col.get("high_cardinality") is True


def test_low_cardinality_no_flag():
    p = profile_dataframe(MIXED)
    city = next(c for c in p["columns"] if c["name"] == "city")
    assert not city.get("high_cardinality")


def test_text_high_cardinality_warns():
    text = profile_to_text(profile_dataframe(HIGH_CARD))
    assert "high-cardinality" in text
    assert "not suitable for charts" in text


def test_text_low_cardinality_no_warning():
    text = profile_to_text(profile_dataframe(MIXED))
    assert "high-cardinality" not in text
