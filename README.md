# Autonomous Data Analyst

A Streamlit app that takes a CSV or Excel file, automatically decides which charts are most insightful, generates them, and explains the findings in plain language. Built on LangGraph with tool-calling so the agent can write and execute its own analysis code.

## How it works

1. **Profile** — the dataset is profiled (dtypes, null %, cardinality, stats, datetime ranges)
2. **Plan** — an LLM picks the most relevant chart types based on the profile
3. **Explain** — a second LLM pass writes a plain-language insight for each chart before any code runs
4. **Execute** — a ReAct agent writes and runs pandas/matplotlib/plotly code in a sandboxed REPL; figures are captured automatically on `plt.show()` / `save_figure(fig)`
5. **Chat** — the user can ask follow-up questions; the agent retains full conversation context

## Stack

- **UI** — Streamlit
- **Agent** — LangGraph (planner → explainer → agent ↔ tools → harvester loop)
- **LLM** — Anthropic Claude or OpenAI (configurable via `.env`)
- **Charts** — matplotlib (PNG) and plotly (interactive HTML)

## Setup

**1. Install dependencies**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**2. Configure your LLM provider**

Copy `.env.example` to `.env` and fill in your API key:

```env
# Anthropic (Claude)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...

# — or — OpenAI
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o
# OPENAI_API_KEY=sk-...
```

**3. Run**

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a CSV or Excel file, and click **Analyze automatically**.

## Sample data

```bash
python create_sample_data.py   # generates sample_data.xlsx (200 rows, sales data)
```

## Project structure

```
src/
  ingestion.py      CSV / Excel parser
  profiler.py       Dataset profiler
  chart_planner.py  LLM chart-decision node
  explainer.py      LLM chart-explanation node
  chart_tools.py    Python REPL with matplotlib / plotly capture
  agent.py          LangGraph graph (planner → explainer → agent ↔ tools)
  state.py          AgentState TypedDict
  llm.py            LLM provider factory
tests/              79 unit tests
app.py              Streamlit UI
```
