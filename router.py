import os
import json
import requests
from typing import TypedDict, Optional
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch

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
    trim_memory,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
   api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", r"""You are a chemistry assistant. Given the user's input, extract two things:
1. intent: one of 'pubchem', 'descriptor', 'pdb', 'smallworld', 'patent', 'web', 'visualize', or 'fallback'
2. compound: a chemical name or identifier (e.g., aspirin, caffeine, ibuprofen, 1HVR for PDB)

Intent Descriptions:
- pubchem: For looking up molecular structure (SMILES, molecular formula) from common chemical names or identifiers using PubChem.
  Example: "What is the structure of aspirin?"

- descriptor: For computing molecular descriptors such as molecular weight, TPSA, or LogP based on a known compound.
  Example: "What’s the molecular weight of caffeine?"

- pdb: For retrieving metadata from the Protein Data Bank (PDB), given a PDB ID (e.g., 1HVR).
  Example: "Tell me the resolution of 1HVR."

- smallworld: For finding similar molecules using SmallWorld search.
  Example: "Find molecules similar to ibuprofen."

- patent: For querying chemical-related patents using Google Patents or USPTO API.
  Example: "Are there any patents related to GLP-1?"

- visualize: For drawing and displaying the 2D molecular structure from the SMILES of a compound.
  Example: "Show me what caffeine looks like."

- web: For any general-purpose questions that require web search, including scientific trends, research developments, or informational queries not tied to a specific compound or ID.
    Example: "What’s the latest trend on aspirin?"

- fallback: Use this when the input does not match any of the above categories. Double check if any of the intents can be applied, you should not return this intent unless you are sure that the input does not match any of the intents.

Examples:
Input: What is the structure of aspirin?
Output: {{ "intent": "pubchem", "compound": "aspirin" }}

Input: What’s the molecular weight of caffeine?
Output: {{ "intent": "descriptor", "compound": "caffeine" }}

Input: Tell me the resolution of 1HVR.
Output: {{ "intent": "pdb", "compound": "1HVR" }}

Input: Find molecules similar to ibuprofen.
Output: {{ "intent": "smallworld", "compound": "ibuprofen" }}

Input: Are there any patents related to GLP-1?
Output: {{ "intent": "patent", "compound": "GLP-1" }}

Input: What is aspirin used for?
Output: {{ "intent": "web", "compound": "aspirin" }}


Input: visualize caffeine.
Output: {{ "intent": "visualize", "compound": "caffeine" }}



Note: it is possilbe that the user ask multiple question with same intent, e.g. "What is the structure of aspirin? What is the structure of ibuprofen?"

Return JSON in the format: {{ "intent": "descriptor", "compound": "aspirin" }}

If the compound is not explicitly mentioned in the input, and the input appears to be a follow-up (e.g. "What about its LogP?"), then return "compound": null to indicate it should be inferred from previous memory.
Do not extract parts of the question as the compound name.
"""
    ),
    ("human", "{input}")
])

router_chain = prompt | llm


async def router_node(state: AgentState):
    input_text = state["input"]
    try:
        result = router_chain.invoke({"input": input_text}).content
        print(f"[Router Node] Prompt Input: {input_text}")
        print(f"[Router Output] Raw LLM response: {result}")

        parsed = json.loads(result)
        compound = parsed.get("compound", None)

        # Try to recover compound from memory if missing
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

        intent = parsed.get("intent", "fallback") or "fallback"
        intent = intent.lower()

        if intent in {"pubchem", "descriptor", "smallworld", "visualize"} and compound:
            smiles, formula = await get_smiles_from_pubchem(compound)
        else:
            smiles, formula = None, None

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
        elif "calc" in intent:
            next_node = "calc_node"
        elif "visualize" in intent:
            next_node = "visualize_node"
        else:
            next_node = "fallback_node"

        state["intent"] = intent

        return {
            "__next__": next_node,
            "input": input_text,
            "memory": state.get("memory", []),
            "compound": compound,
            "smiles": smiles,
            "formula": formula,
            "descriptors": None,
            "pdb_metadata": None,
            "intent": intent
        }

    except Exception as e:
        print(f"[Router Node] Exception: {e}")
        return {
            "__next__": "fallback_node",
            **state
        }