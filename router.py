import re
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Optional
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
# LLM-based router imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from serpapi import GoogleSearch
import json

from utils import (
    get_smiles_from_pubchem,
    get_molecular_properties,
    get_pdb_metadata,
    smallworld_submit_search,
    smallworld_view_results,
    query_google_patents_via_serpapi,
    format_patent_results,
    log_state,
    AgentState
    , trim_memory)


llm = ChatOpenAI(model="gpt-3.5-turbo",
                 api_key="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")  # Use your preferred model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chemistry assistant. Given the user's input, extract two things:\n"
               "1. intent: one of 'pubchem', 'descriptor', 'pdb', 'smallworld', 'patent', 'web', or 'fallback'\n"
               "2. compound: a chemical name or identifier (e.g., aspirin, caffeine, ibuprofen, 1HVR for PDB)\n\n"
               "Descriptions:\n"
               "- pubchem: For molecular structure lookup from names using PubChem.\n"
               "- descriptor: For calculating molecular weight, TPSA, or logP from SMILES.\n"
               "- pdb: For protein metadata from the Protein Data Bank using PDB ID.\n"
               "- smallworld: For finding similar molecules using the SmallWorld API.\n"
               "- patent: For querying chemistry-related patents using SerpAPI.\n"
               "- web: For general web search about chemical concepts, use, mechanism, etc.\n"
               "- fallback: Use this when no intent or compound can be identified.\n\n"
               "Return JSON in the format: {{\"intent\": intent, \"compound\": compound_or_none}}"),
    ("human", "{input}")
])
router_chain = prompt | llm

# Explicit router node that routes and mutates state
def router_node(state: AgentState):
    input_text = state["input"]
    try:
        result = router_chain.invoke({"input": input_text}).content
        print(f"[Router Output] Raw LLM response: {result}")
        parsed = json.loads(result)
        compound = parsed.get("compound", None)
        # Fallback: Try to recover compound from memory
        if not compound:
            for msg in reversed(state.get("memory", [])):
                if isinstance(msg, AIMessage) and "SMILES" in msg.content:
                    match = re.search(r"^(.*?)\s*→\s*SMILES:", msg.content)
                    if match:
                        compound = match.group(1).strip().lower()
                        break
                elif isinstance(msg, HumanMessage) and "aspirin" in msg.content.lower():
                    compound = "aspirin"
                    break
        intent = parsed.get("intent", "fallback")
        if intent is None:
            intent = "fallback"
        intent = intent.lower()
        # For these intents, try to look up SMILES and formula
        if intent in {"pubchem", "descriptor", "smallworld"} and compound:
            smiles, formula = get_smiles_from_pubchem(compound)
        else:
            smiles, formula = None, None
        # Compose next node
        if "pubchem" in intent:
            next_node = "pubchem_node"
        elif "descriptor" in intent or "property" in intent:
            next_node = "descriptor_node"
        elif "pdb" in intent or "protein" in intent:
            next_node = "pdb_node"
        elif "smallworld" in intent or "similar" in intent:
            next_node = "smallworld_node"
        elif "patent" in intent:
            next_node = "patent_node"
        elif "web" in intent or "search" in intent or "find" in intent:
            next_node = "web_node"
        else:
            next_node = "fallback_node"
        return {
            "__next__": next_node,
            "input": input_text,
            "memory": state.get("memory", []),
            "compound": compound,
            "smiles": smiles,
            "formula": formula,
            "descriptors": None,
            "pdb_metadata": None
        }
    except Exception as e:
        print(f"[Router Node] Exception: {e}")
        # Fallback, merge with prior state
        return {
            "__next__": "fallback_node",
            **state
        }