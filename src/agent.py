import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.state import AgentState
from src.tools import create_data_tool

# Load environment variables 
load_dotenv()

def create_agent_graph(df: pd.DataFrame):
    """
    Builds and compiles the LangGraph state machine for the Data Analyst Agent.
    """
    # 1. Initialize the tool with the user's DataFrame
    data_tool = create_data_tool(df)
    tools = [data_tool]

    # 2. Initialize the LLM (temperature=0 ensures deterministic, focused code generation)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Bind the tools to the LLM so it knows what actions it can take
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define the System Prompt
    system_message = """You are an autonomous Senior Data Analyst.
    You have access to a pandas DataFrame named 'df'.
    Your job is to answer user queries by writing and executing Python/Pandas code.
    
    Instructions:
    - Always execute code to find the answer; do not guess.
    - Use the 'python_repl' tool to run your pandas code.
    - Always use print() to output the results you need to see.
    - If you encounter an error, analyze it, correct your code, and try again.
    - Once you have the final answer, explain it clearly to the user.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Create the chain: Prompt -> LLM (with tools)
    agent_chain = prompt | llm_with_tools

    # 4. Define the Graph Nodes
    def agent_node(state: AgentState):
        """Invokes the LLM to decide the next step or formulate an answer."""
        result = agent_chain.invoke({"messages": state["messages"]})
        return {"messages": [result]}

    # ToolNode automatically handles the execution of the tools requested by the LLM
    tool_node = ToolNode(tools)

    # 5. Build the Graph Workflow
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set the starting point
    workflow.set_entry_point("agent")

    # Add conditional routing:
    # If the agent calls a tool -> go to 'tools' node.
    # If the agent finishes its logic -> go to END.
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )

    # After a tool finishes executing, always route back to the agent to read the results
    workflow.add_edge("tools", "agent")

    # Compile the graph into a runnable application
    app = workflow.compile()

    return app