import io
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from langchain_core.messages import HumanMessage

from src.ingestion import load_file, IngestionError
from src.agent import create_agent_graph

_AUTO_PROMPT = (
    "Analyze this dataset: generate the recommended charts and explain the key insights."
)

_CSS = """
<style>
/* Subtle download buttons — should not compete visually with the chat */
[data-testid="stDownloadButton"] > button {
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #666;
    background: transparent;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin-top: 0.1rem;
    margin-bottom: 0.6rem;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #999;
    color: #111;
    background: #f5f5f5;
}
/* Tighter gap between image and its download button */
[data-testid="stImage"] { margin-bottom: 0; }
</style>
"""


@st.cache_data(show_spinner=False)
def _parse_upload(data: bytes, filename: str) -> pd.DataFrame:
    return load_file(io.BytesIO(data), filename)


def _init_agent(df: pd.DataFrame, file_key: str) -> None:
    """Build the agent graph once per uploaded file."""
    if st.session_state.get("agent_file_key") == file_key:
        return
    app, p_text = create_agent_graph(df)
    st.session_state.update({
        "agent_file_key":     file_key,
        "agent_app":          app,
        "profile_text":       p_text,
        "chart_plan":         [],
        "chart_explanations": [],
        "lg_messages":        [],
        "chat_history":       [],
    })


def _run_turn(user_input: str) -> None:
    app     = st.session_state["agent_app"]
    lg_msgs = list(st.session_state["lg_messages"]) + [HumanMessage(content=user_input)]

    try:
        result = app.invoke({
            "messages":           lg_msgs,
            "profile_text":       st.session_state["profile_text"],
            "chart_plan":         st.session_state["chart_plan"],
            "chart_explanations": st.session_state["chart_explanations"],
            "captured_figures":   [],
        })
    except Exception as exc:
        st.session_state["chat_history"].append({"role": "user",      "content": user_input,                              "figures": []})
        st.session_state["chat_history"].append({"role": "assistant", "content": f"⚠️ Something went wrong: {exc}\n\nPlease try again.", "figures": []})
        return

    st.session_state["lg_messages"]        = result["messages"]
    st.session_state["chart_plan"]         = result.get("chart_plan", [])
    st.session_state["chart_explanations"] = result.get("chart_explanations", [])

    last_msg = result["messages"][-1]
    ai_text  = last_msg.content if isinstance(last_msg.content, str) else ""
    figures  = result.get("captured_figures", [])

    st.session_state["chat_history"].append({"role": "user",      "content": user_input, "figures": []})
    st.session_state["chat_history"].append({"role": "assistant", "content": ai_text,    "figures": figures})


def _render_figures(figures: list[dict], key_prefix: str) -> None:
    for i, fig in enumerate(figures):
        if fig["format"] == "png":
            st.image(io.BytesIO(fig["data"]))
            st.download_button(
                label="⬇ Download PNG",
                data=fig["data"],
                file_name=f"chart_{i + 1}.png",
                mime="image/png",
                key=f"{key_prefix}_png_{i}",
            )
        elif fig["format"] == "html":
            components.html(fig["data"], height=450, scrolling=False)
            st.download_button(
                label="⬇ Download HTML",
                data=fig["data"],
                file_name=f"chart_{i + 1}.html",
                mime="text/html",
                key=f"{key_prefix}_html_{i}",
            )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analyst", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)
st.title("Autonomous Data Analyst")
st.caption("Upload a spreadsheet and the AI will decide which charts to generate and explain them.")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your file",
    type=["xlsx", "xls", "csv"],
    help="Excel (.xlsx / .xls) or CSV files are supported.",
)

if uploaded is None:
    st.divider()
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown(
            "### Get started\n"
            "1. Upload a CSV or Excel file above\n"
            "2. Click **🔍 Analyze automatically** to generate charts and insights\n"
            "3. Ask follow-up questions in the chat below"
        )
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("Reading file…"):
    try:
        df = _parse_upload(uploaded.getvalue(), uploaded.name)
    except IngestionError as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error reading '{uploaded.name}': {exc}")
        st.stop()

rows, cols = df.shape
st.success(f"Loaded **{rows:,} rows × {cols} columns** from `{uploaded.name}`")

with st.expander("Data preview", expanded=False):
    st.dataframe(df.head(5), use_container_width=True)

with st.expander("Column summary", expanded=False):
    summary = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "non-null": df.notna().sum(),
        "null %":   (df.isna().mean() * 100).round(1).astype(str) + "%",
        "unique":   df.nunique(),
    })
    st.dataframe(summary, use_container_width=True)

# ── Agent ─────────────────────────────────────────────────────────────────────
_init_agent(df, file_key=f"{uploaded.name}:{uploaded.size}")

# ── Chat history ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("💬 Analysis")

for t_idx, turn in enumerate(st.session_state.get("chat_history", [])):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        _render_figures(turn.get("figures", []), key_prefix=str(t_idx))

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask something about your data…")

if not st.session_state.get("chat_history"):
    if st.button("🔍 Analyze automatically", type="primary"):
        with st.spinner("Analyzing dataset…"):
            _run_turn(_AUTO_PROMPT)
        st.rerun()

if user_input:
    with st.spinner("Thinking…"):
        _run_turn(user_input)
    st.rerun()
