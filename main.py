import os
import asyncio
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

from utils import (
    get_smiles_from_pubchem,
    get_molecular_properties,
    get_pdb_metadata,
    smallworld_submit_search,
    smallworld_view_results,
    query_google_patents_via_serpapi,
    format_patent_results,
    log_state,
    AgentState,
    trim_memory
)

from nodes import (
    descriptor_node,
    pubchem_node,
    smallworld_node,
    pdb_node,
    patent_node,
    web_node,
    fallback_node,
    calc_node,
    visualize_node
)

from router import router_node

load_dotenv()

# ------------------
# Graph Wiring
# ------------------

graph = StateGraph(AgentState)
graph.add_node("pubchem_node", pubchem_node)
graph.add_node("descriptor_node", descriptor_node)
graph.add_node("pdb_node", pdb_node)
graph.add_node("smallworld_node", smallworld_node)
graph.add_node("patent_node", patent_node)
graph.add_node("web_node", web_node)
graph.add_node("fallback_node", fallback_node)
graph.add_node("calc_node", calc_node)
graph.add_node("visualize_node", visualize_node)

# Add explicit router node
graph.add_node("router", router_node)
graph.set_entry_point("router")

# Add conditional edges from router node for intent-based routing (older-compatible lambda version)
graph.add_conditional_edges(
    "router",
    lambda state: state["__next__"],
    {
        "descriptor_node": "descriptor_node",
        "pubchem_node": "pubchem_node",
        "pdb_node": "pdb_node",
        "smallworld_node": "smallworld_node",
        "patent_node": "patent_node",
        "web_node": "web_node",
        "fallback_node": "fallback_node",
        "calc_node": "calc_node",
        "visualize_node": "visualize_node",
    }
)
graph.set_finish_point("descriptor_node")
graph.set_finish_point("pubchem_node")
graph.set_finish_point("pdb_node")
graph.set_finish_point("smallworld_node")
graph.set_finish_point("patent_node")
graph.set_finish_point("web_node")
graph.set_finish_point("fallback_node")
graph.set_finish_point("calc_node")
graph.set_finish_point("visualize_node")

agent = graph.compile()

# Chat history variable
chat_history = []

# Helper function to invoke the agent and manage memory
async def chat(input_text: str):
    global chat_history
    # Extract latest metadata if available
    last_state = chat_history[-1].additional_kwargs.get("state", {}) if chat_history else {}
    state = {
        "input": input_text,
        "memory": chat_history,
        "compound": last_state.get("compound", None),
        "smiles": last_state.get("smiles", None),
        "formula": last_state.get("formula", None),
        "descriptors": last_state.get("descriptors", None),
        "pdb_metadata": last_state.get("pdb_metadata", None)
    }
    result = await agent.ainvoke(state)
    print(f"[Chat] Input: {input_text} — Result: {result}")
    # Embed latest state in the response metadata
    if hasattr(result["memory"][-1], "additional_kwargs"):
        result["memory"][-1].additional_kwargs["state"] = {
            "compound": result.get("compound"),
            "smiles": result.get("smiles"),
            "formula": result.get("formula"),
            "descriptors": result.get("descriptors"),
            "pdb_metadata": result.get("pdb_metadata")
        }
    chat_history = result["memory"]
    return result["memory"][-1].content if result["memory"] else "No response generated."


# # Usage example
# if __name__ == "__main__":
#     print("ChemAgent is ready. Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in {"exit", "quit"}:
#             print("Goodbye!")
#             break
#         response = asyncio.run(chat(user_input))
#         print("Assistant:", response)
