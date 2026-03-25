import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool

def create_data_tool(df: pd.DataFrame) -> PythonAstREPLTool:
    """
    Creates a Python REPL tool with the DataFrame injected into its local environment.
    This allows the LLM to execute Pandas code assuming 'df' already exists.
    """
    # Inject 'df' and 'pd' into the locals dictionary so the tool recognizes them
    tool = PythonAstREPLTool(locals={"df": df, "pd": pd})
    tool.name = "python_repl"
    tool.description = (
        "A Python shell. Use this to execute Python and Pandas commands. "
        "Input should be a valid Python command. "
        "IMPORTANT: The user's dataset is already loaded and available as a pandas DataFrame named 'df'. "
        "To see the output of a value, make sure to use print()."
    )
    return tool