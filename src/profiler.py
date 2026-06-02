from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Produce a structured profile of a DataFrame for LLM consumption.

    Returns a dict with:
      - shape: (rows, cols)
      - columns: list of per-column dicts with dtype, null_pct, cardinality,
                 stats (numeric) or top_values (categorical), and sample values
    """
    rows, cols = df.shape
    columns = []

    for col in df.columns:
        series = df[col]
        null_pct = round(series.isna().mean() * 100, 1)
        cardinality = int(series.nunique(dropna=True))
        sample = series.dropna().head(3).tolist()

        entry: dict = {
            "name": col,
            "dtype": _friendly_dtype(series),
            "null_pct": null_pct,
            "cardinality": cardinality,
            "sample": sample,
        }

        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            entry["stats"] = {
                "min": _fmt(described.get("min")),
                "max": _fmt(described.get("max")),
                "mean": _fmt(described.get("mean")),
                "std": _fmt(described.get("std")),
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            min_val, max_val = series.min(), series.max()
            entry["range"] = {
                "min": str(min_val.date()) if pd.notna(min_val) else "n/a",
                "max": str(max_val.date()) if pd.notna(max_val) else "n/a",
            }
        else:
            top = (
                series.value_counts(dropna=True)
                .head(5)
                .index.tolist()
            )
            entry["top_values"] = top
            entry["high_cardinality"] = rows > 10 and (cardinality / rows) > 0.5

        columns.append(entry)

    return {"shape": {"rows": rows, "cols": cols}, "columns": columns}


def profile_to_text(profile: dict) -> str:
    """
    Render a profile dict as a compact text block suitable for an LLM prompt.
    """
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    lines = [f"Dataset: {rows} rows × {cols} columns"]

    for c in profile["columns"]:
        line = f"- {c['name']} ({c['dtype']}): {c['null_pct']}% null, {c['cardinality']} unique"
        if "stats" in c:
            s = c["stats"]
            line += f" | min={s['min']} max={s['max']} mean={s['mean']}"
        elif "range" in c:
            r = c["range"]
            line += f" | range: {r['min']} → {r['max']}"
        else:
            tops = ", ".join(str(v) for v in c["top_values"][:3])
            line += f" | top: {tops}"
            if c.get("high_cardinality"):
                line += " [high-cardinality — not suitable for charts]"
        lines.append(line)

    return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _friendly_dtype(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _fmt(val) -> str:
    if val is None:
        return "n/a"
    try:
        return str(round(float(val), 4))
    except (TypeError, ValueError):
        return str(val)
