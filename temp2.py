


# Web search dependencies
import os
from serpapi import GoogleSearch

from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

# Dummy message classes for context (replace with actual imports)
class HumanMessage:
    def __init__(self, content): self.content = content
class AIMessage:
    def __init__(self, content): self.content = content

# Dummy AgentState type (replace with actual)
AgentState = Dict[str, Any]

# Dummy memory helpers (replace with actual implementations)
def trim_memory(memory): return memory
def add_messages(memory, *msgs): return memory + list(msgs)

# --- LLM Router Prompt (add 'web' intent) ---
llm_router_prompt = (
    "You are a chemistry assistant. Based on the user's question, decide whether the intent is one of the following: 'pubchem', 'descriptor', 'pdb', 'smallworld', 'patent', 'web', or 'fallback'. Only respond with the single word."
)

# --- route_tool function (extend for web intent) ---
def route_tool(intent: str):
    intent = intent.lower()
    if "pubchem" in intent:
        return "pubchem_node"
    elif "descriptor" in intent:
        return "descriptor_node"
    elif "pdb" in intent:
        return "pdb_node"
    elif "smallworld" in intent:
        return "smallworld_node"
    elif "patent" in intent:
        return "patent_node"
    elif "web" in intent or "search" in intent or "find" in intent:
        return "web_node"
    else:
        return "fallback_node"

# --- web_node function ---
def web_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return {**state, "memory": add_messages(messages, AIMessage(content="Missing SERPAPI_KEY."))}

    params = {
        "q": input_text,
        "api_key": serpapi_key,
        "engine": "google",
        "num": 5
    }

    try:
        search = GoogleSearch(params)
        result = search.get_dict()
        organic = result.get("organic_results", [])[:5]
        if not organic:
            content = "No web results found."
        else:
            lines = [
                f"{i+1}. {item.get('title', 'N/A')}\n{item.get('link', '')}"
                for i, item in enumerate(organic)
            ]
            content = "Top web search results:\n" + "\n\n".join(lines)
    except Exception as e:
        content = f"Web search error: {str(e)}"

    return {**state, "memory": add_messages(messages, AIMessage(content=content))}

# --- Register node in graph (example) ---
# Assuming you have a graph object, register the node:
# graph.add_node("web_node", web_node)

# --- Extend conditional edges ---
# Add to your node routing dict or logic:
# "web_node": "web_node",

if __name__ == "__main__":
    # Simulate a user input asking for web search
    test_state = {
        "input": "Find recent news about GLP-1 weight loss drugs",
        "memory": []
    }

    # Call the web_node directly
    result = web_node(test_state)

    # Print the assistant's last response
    last_message = result["memory"][-1].content
    print("Assistant Response:\n")
    print(last_message)