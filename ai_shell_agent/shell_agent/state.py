# Import necessary modules
from typing import List
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

# Define the custom state model
class CustomState(BaseModel):
    """
    Custom state model for the AI shell agent.
    """
    messages: List[Annotated[str, HumanMessage], add_messages] = []
    internal_support_messages: List[Annotated[str, AnyMessage], add_messages] = []
    user_os: str = "user_os"
