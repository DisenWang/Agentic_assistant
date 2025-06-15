import os
from typing import TypedDict, Optional

import requests
from rdkit import Chem
from rdkit.Chem import Draw

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
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
    trim_memory
    , add_messages
)

def web_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("Web Node", state)
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

async def smallworld_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("SmallWorld Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    smiles = state.get("smiles")
    if not smiles:
        return {**state, "memory": add_messages(messages, AIMessage(content="Please provide a compound name first before doing similarity search."))}
    try:
        hlid = await smallworld_submit_search(smiles)
        smiles = state.get("smiles")
        results = await smallworld_view_results(hlid, smiles)
        if not isinstance(results, list):
            raise ValueError(f"Unexpected results format: {type(results)} — {results}")
        top_results = results[:5]
        print(f"[SmallWorld Node] Retrieved {len(top_results)} results for SMILES: {smiles}")
        top_lines = [
            f"{i+1}. SMILES: {item.get('smiles', 'N/A')}, ID: {item.get('id', 'N/A')}, Similarity: {item.get('similarity', 'N/A')}"
            for i, item in enumerate(top_results)
        ]
        content = "Top similar molecules:\n" + "\n".join(top_lines)
        return {**state, "memory": add_messages(messages, AIMessage(content=content))}
    except Exception as e:
        return {**state, "memory": add_messages(messages, AIMessage(content=f"SmallWorld search failed: {str(e)}"))}

async def pubchem_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("PubChem Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    compound = state.get("compound") or input_text.strip().split()[-1]
    if not compound:
        return {**state, "memory": add_messages(messages, AIMessage(content="Please specify the compound name."))}
    state["compound"] = compound
    smiles, formula = await get_smiles_from_pubchem(compound)
    if smiles:
        response = f"{compound.capitalize()} → SMILES: {smiles}, Formula: {formula}"
        return {**state, "compound": compound, "smiles": smiles, "formula": formula, "memory": add_messages(messages, AIMessage(content=response))}
    else:
        return {**state, "memory": add_messages(messages, AIMessage(content=f"Sorry, I couldn't find {compound} in the PubChem database. Please check the compound name or try a related synonym."))}

async def descriptor_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("Descriptor Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = memory + [HumanMessage(content=input_text)]
    smiles = state.get("smiles")
    compound = state.get("compound")
    # No fallback extraction for compound; rely on router to set it
    if not smiles and compound:
        smiles, formula = await get_smiles_from_pubchem(compound)
        print(f"[Descriptor Node] Retrieved SMILES: {smiles} for compound: {compound}")
        state["smiles"] = smiles
        state["formula"] = formula
        smiles = state["smiles"]

    if not smiles:
        messages.append(AIMessage(content="Please provide the compound name."))
        return {**state, "memory": messages}
    descriptors = get_molecular_properties(smiles)
    print(f"[Descriptor Node] Final descriptors: {descriptors}")
    input_lower = input_text.lower()

    key_map = {
        "molecular weight": "MolecularWeight",
        "logp": "LogP",
        "tpsa": "TPSA",
        "polar surface area": "TPSA",
    }

    selected = {
        v: descriptors[v]
        for k, v in key_map.items()
        if k in input_lower and v in descriptors
    }

    response = selected if selected else descriptors

    print(f"[Descriptor Node] Memory before return: {messages}")
    messages.append(AIMessage(content=str(response)))
    return {**state, "descriptors": descriptors, "memory": messages}

def pdb_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("PDB Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    match = re.search(r"(?:pdb\s+id\s*|pdb\s*[:=]?\s*|protein\s+)(\w+)", input_text.lower())
    pdb_id = match.group(1) if match else None
    if not pdb_id:
        return {**state, "memory": add_messages(messages, AIMessage(content="Please provide a valid PDB ID."))}
    metadata = get_pdb_metadata(pdb_id)
    return {**state, "pdb_metadata": metadata, "memory": add_messages(messages, AIMessage(content=str(metadata)))}

def patent_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("Patent Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    try:
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            raise ValueError("SERPAPI_KEY not set in environment.")
        data = query_google_patents_via_serpapi(input_text, serpapi_key)
        if isinstance(data, str):
            content = f"Patent search failed: {data}"
        else:
            content = format_patent_results(data)
    except Exception as e:
        content = f"Patent search error: {str(e)}"
    return {**state, "memory": add_messages(messages, AIMessage(content=content))}

def fallback_node(state: AgentState) -> AgentState:
    memory = trim_memory(state.get("memory", []))
    input_text = state["input"]
    log_state("Fallback Node", state)
    messages = add_messages(memory, HumanMessage(content=input_text))
    return {**state, "memory": add_messages(messages, AIMessage(content="Sorry, I didn't understand your question. Try asking about SMILES, molecular weight, LogP, or protein structures."))}

def calc_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("Calc Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))
    # Placeholder logic for calculation
    result = "This is a placeholder for chemistry calculation results."
    messages = add_messages(messages, AIMessage(content=result))
    return {**state, "memory": messages}

def visualize_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    log_state("Visualize Node", state)
    memory = trim_memory(state.get("memory", []))
    messages = add_messages(memory, HumanMessage(content=input_text))

    smiles = state.get("smiles")
    if not smiles:
        messages = add_messages(messages, AIMessage(content="SMILES not found in state. Cannot generate visualization."))
        return {**state, "memory": messages}

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        output_path = os.path.join(os.getcwd(), "output.png")
        img.save(output_path)
        result = "Molecule image saved to output.png"
    else:
        result = "Failed to generate molecule from SMILES."

    messages = add_messages(messages, AIMessage(content=result))
    return {**state, "memory": messages, "intent": "visualize"}