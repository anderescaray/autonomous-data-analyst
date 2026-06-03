"""File ingestion — parse CSV and Excel uploads into a clean DataFrame."""
import io
from pathlib import Path

import pandas as pd


class IngestionError(ValueError):
    pass


def load_file(source: str | Path | io.IOBase, filename: str = "") -> pd.DataFrame:
    """
    Parse an Excel or CSV file into a DataFrame.

    Args:
        source:   File path, Path object, or binary IO (e.g. Streamlit UploadedFile).
        filename: Original filename — used to infer format when source is a buffer.

    Returns:
        A clean DataFrame with stripped column names.

    Raises:
        IngestionError: If the file cannot be parsed or produces an empty result.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    ext = Path(name).suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = _read_excel(source, name)
    elif ext == ".csv":
        df = _read_csv(source, name)
    else:
        # Try both parsers before giving up
        df = _try_both(source, name)

    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise IngestionError("The file was parsed but contains no data.")

    return df


# ── private helpers ────────────────────────────────────────────────────────────

def _read_excel(source, name: str) -> pd.DataFrame:
    """Read an Excel file via openpyxl, raising IngestionError on failure."""
    try:
        return pd.read_excel(source, engine="openpyxl")
    except Exception as exc:
        raise IngestionError(f"Could not read Excel file '{name}': {exc}") from exc


def _read_csv(source, name: str) -> pd.DataFrame:
    """Try common encodings and separators; keep the parse with the most columns."""
    import pandas.errors

    best: pd.DataFrame | None = None

    for encoding in ("utf-8", "latin-1", "utf-8-sig"):
        for sep in (",", ";", "\t"):
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                candidate = pd.read_csv(source, sep=sep, encoding=encoding)
                # Keep the parse that produces the most columns (best separator match)
                if best is None or len(candidate.columns) > len(best.columns):
                    best = candidate
            except pandas.errors.EmptyDataError:
                raise IngestionError("The file was parsed but contains no data.")
            except Exception:
                continue

    if best is None:
        raise IngestionError(f"Could not read CSV file '{name}' — tried utf-8/latin-1 × ,/;/tab.")

    return best


def _try_both(source, name: str) -> pd.DataFrame:
    """Fallback when the extension is missing or unknown."""
    for reader in (_read_excel, _read_csv):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return reader(source, name)
        except IngestionError as exc:
            if "no data" in str(exc):
                raise
            continue
    raise IngestionError(
        f"Could not determine file format for '{name}'. "
        "Please upload a .xlsx, .xls, or .csv file."
    )
