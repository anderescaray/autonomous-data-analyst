import io
import pandas as pd
import pytest

from src.ingestion import load_file, IngestionError


def _make_excel(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def _make_csv(df: pd.DataFrame, sep=",", encoding="utf-8") -> io.BytesIO:
    return io.BytesIO(df.to_csv(index=False, sep=sep).encode(encoding))


SAMPLE = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25], "score": [88.5, 92.0]})


# ── happy paths ───────────────────────────────────────────────────────────────

def test_load_excel():
    df = load_file(_make_excel(SAMPLE), "data.xlsx")
    assert list(df.columns) == ["name", "age", "score"]
    assert len(df) == 2


def test_load_csv_comma():
    df = load_file(_make_csv(SAMPLE), "data.csv")
    assert list(df.columns) == ["name", "age", "score"]


def test_load_csv_semicolon():
    df = load_file(_make_csv(SAMPLE, sep=";"), "data.csv")
    assert list(df.columns) == ["name", "age", "score"]


def test_load_csv_latin1():
    data = pd.DataFrame({"ciudad": ["Córdoba", "Málaga"]})
    df = load_file(_make_csv(data, encoding="latin-1"), "data.csv")
    assert len(df) == 2


def test_strips_column_whitespace():
    dirty = pd.DataFrame({" name ": ["X"], "  age": [1]})
    buf = _make_excel(dirty)
    df = load_file(buf, "dirty.xlsx")
    assert "name" in df.columns
    assert "age" in df.columns


def test_fallback_no_extension():
    df = load_file(_make_csv(SAMPLE), "upload")
    assert len(df) == 2


# ── error paths ───────────────────────────────────────────────────────────────

def test_empty_file_raises():
    with pytest.raises(IngestionError, match="no data"):
        load_file(io.BytesIO(b""), "empty.csv")


def test_empty_file_no_extension_raises():
    with pytest.raises(IngestionError, match="no data"):
        load_file(io.BytesIO(b""), "upload")


def test_garbage_raises():
    with pytest.raises(IngestionError):
        load_file(io.BytesIO(b"\x00\x01\x02\x03"), "bad.xlsx")
