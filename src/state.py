from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state (memory) of our agent as it progresses through the graph.
    """
    # 'add_messages' ensures that new messages are appended to the list, rather than overwriting it
    messages: Annotated[list[BaseMessage], add_messages]