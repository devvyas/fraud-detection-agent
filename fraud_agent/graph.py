from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from fraud_agent.tools import get_spend_history, get_travel_signals, get_device_fingerprint

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# State is the shared notepad passed between every node in the graph.
# `add_messages` is a reducer: instead of replacing the list on each step,
# it *appends* new messages. This is what gives the LLM its memory of prior
# tool calls within one investigation run.

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------
_TOOLS = [get_spend_history, get_travel_signals, get_device_fingerprint]

_llm = ChatGroq(model="llama-3.3-70b-versatile")

# bind_tools tells the LLM which tools exist and what their schemas are.
# The LLM will emit a ToolCall message (not text) when it wants to use one.
_llm_with_tools = _llm.bind_tools(_TOOLS)

_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a fraud investigation agent. "
    "You have tools to check spending history, travel signals, and device fingerprints. "
    "Investigate efficiently — only call tools that will meaningfully change your "
    "understanding of whether this transaction is suspicious. "
    "If early evidence is already conclusive, stop early. "
    "When you have enough evidence, summarize your findings in plain text. "
    "Do NOT make a block/approve decision yourself — a separate scoring system will decide."
))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def call_llm(state: State) -> dict:
    """The agent node: sends all current messages to the LLM and gets back
    either a tool call or a final text response."""
    response = _llm_with_tools.invoke(state["messages"])
    # Returning a dict with 'messages' triggers the add_messages reducer,
    # which appends the response to the existing list.
    return {"messages": [response]}


# ToolNode is a prebuilt node that reads the last message, sees which tool
# the LLM called, runs it, and appends a ToolMessage with the result.
_tool_node = ToolNode(_TOOLS)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def should_continue(state: State) -> str:
    """Inspect the last message. If the LLM emitted tool calls, route to the
    tool node. Otherwise the LLM is done — route to END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    # Add nodes — each node is a callable that receives State and returns a dict
    graph.add_node("agent", call_llm)
    graph.add_node("tools", _tool_node)

    # Entry point: always start at the agent node
    graph.set_entry_point("agent")

    # Conditional edge: after the agent runs, ask should_continue where to go
    graph.add_conditional_edges("agent", should_continue)

    # Unconditional edge: after tools run, always go back to the agent
    # This creates the loop: agent → tools → agent → tools → ... → END
    graph.add_edge("tools", "agent")

    return graph.compile()
