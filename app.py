import streamlit as st
import pandas as pd

from src.ingestion import load_file, IngestionError

st.set_page_config(page_title="Data Analyst", layout="wide")

st.title("Autonomous Data Analyst")
st.caption("Upload a spreadsheet and the AI will decide which charts to generate and explain them.")

uploaded = st.file_uploader(
    "Upload your file",
    type=["xlsx", "xls", "csv"],
    help="Excel (.xlsx / .xls) or CSV files are supported.",
)

if uploaded is None:
    st.info("Upload a file above to get started.")
    st.stop()

# ── Parse ──────────────────────────────────────────────────────────────────────
with st.spinner("Reading file…"):
    try:
        df = load_file(uploaded, uploaded.name)
    except IngestionError as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

# ── Store in session so downstream pages can access it ────────────────────────
st.session_state["df"] = df

# ── Preview ───────────────────────────────────────────────────────────────────
rows, cols = df.shape
st.success(f"Loaded **{rows:,} rows × {cols} columns** from `{uploaded.name}`")

with st.expander("Data preview", expanded=True):
    st.dataframe(df.head(5), use_container_width=True)

with st.expander("Column summary"):
    summary = pd.DataFrame({
        "dtype":       df.dtypes.astype(str),
        "non-null":    df.notna().sum(),
        "null %":      (df.isna().mean() * 100).round(1).astype(str) + "%",
        "unique":      df.nunique(),
    })
    st.dataframe(summary, use_container_width=True)
