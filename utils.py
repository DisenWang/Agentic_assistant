import re
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Optional
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, DataStructs
# LLM-based router imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from serpapi import GoogleSearch
import json

load_dotenv()

db='REAL-Database-22Q1'

class AgentState(TypedDict):
    input: str
    memory: list
    compound: Optional[str]
    smiles: Optional[str]
    formula: Optional[str]
    descriptors: Optional[dict]
    pdb_metadata: Optional[dict]


def trim_memory(memory, max_pairs=4):
    max_messages = max_pairs * 2
    return memory[-max_messages:]

# ------------------
# Utility Functions
# ------------------

def get_smiles_from_pubchem(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,MolecularFormula/JSON"
    response = requests.get(url)
    print(f"[PubChem] URL: {url} — Status: {response.status_code}")
    if response.status_code == 200:
        try:
            props = response.json()["PropertyTable"]["Properties"][0]
            print(props)
            return props["CanonicalSMILES"], props["MolecularFormula"]
        except Exception as e:
            print(f"[PubChem] Parsing error: {e}")
            return None, None
    else:
        print(f"[PubChem] Failed: {response.text}")
        return None, None

def get_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

def get_pdb_metadata(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Error: {response.status_code} – invalid PDB ID or not found"

    data = response.json()
    info = {
        "Title": data.get("struct", {}).get("title", "N/A"),
        "Organism": data.get("rcsb_entry_container_identifiers", {}).get("entity_ids", ["N/A"]),
        "Resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", ["N/A"]),
        "Deposition Date": data.get("rcsb_accession_info", {}).get("deposit_date", "N/A"),
        "Release Date": data.get("rcsb_accession_info", {}).get("initial_release_date", "N/A")
    }
    return info


def smallworld_submit_search(smiles: str, db: str = db, dist: int = 4, top: int = 5):
    url = "https://sw.docking.org/search/submit"
    params = {
        "smi": [smiles],
        "db": db,
        "dist": str(dist),
        "top": str(top)
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"Submit error {resp.status_code}: {resp.text}")
    
    # Parse server-sent event (SSE) style data
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            data = json.loads(line[5:])
            if "hlid" in data and data["status"] == "END":
                return data["hlid"]
    raise Exception("HLID not found in response")

def smallworld_view_results(hlid: int, state_smiles: str):
    url = "https://sw.docking.org/search/view"
    params = {
        "hlid": hlid,
        "fmt": "tsv",
        "scores": "graph"
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"View error {resp.status_code}: {resp.text}")
    
    lines = resp.text.strip().splitlines()
    print(f"[SmallWorld View] Raw TSV lines: {lines}")

    results = []
    for line in lines:
        print(f"[Line Raw] {line}")
        if line.strip().lower().startswith("smiles"):
            continue  # Skip header
        parts = line.strip().split("\t")
        print(f"[Line Split] {parts}")
        if len(parts) < 2:
            print("[Skipped] Line does not have 2 parts.")
            continue
        smiles_and_id = parts[0]
        try:
            smiles, zid = smiles_and_id.rsplit(" ", 1)
        except ValueError:
            smiles, zid = smiles_and_id, "N/A"
        try:
            distance = float(parts[1])
            # similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            try:
                # Calculate Tanimoto similarity to the original query molecule
                query_mol = Chem.MolFromSmiles(state_smiles)
                result_mol = Chem.MolFromSmiles(smiles)
                query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
                result_fp = AllChem.GetMorganFingerprintAsBitVect(result_mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(query_fp, result_fp)
            except Exception as e:
                print(f"[Similarity Error] Failed for {smiles}: {e}")
                similarity = 0.0
        except ValueError:
            distance = None
            similarity = 0.0
        results.append({
            "id": zid,
            "smiles": smiles,
            "similarity": similarity
        })
    print(f"[SmallWorld View] Parsed {len(results)} results")
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:5]

def query_google_patents_via_serpapi(query: str, serpapi_key: str, num_results: int = 10):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": serpapi_key,
        "num": num_results
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code} – {response.text}"
    return response.json()

def format_patent_results(results: dict) -> str:
    lines = []
    for i, p in enumerate(results.get("organic_results", []), 1):
        lines.append(f"{i}. {p.get('title', 'N/A')}")
        lines.append(f"  Inventor: {p.get('inventor', 'N/A')}")
        lines.append(f"  Filing Date: {p.get('filing_date', 'N/A')}")
        lines.append(f"  Publication Date: {p.get('publication_date', 'N/A')}")
        lines.append(f"  Link: {p.get('patent_link', 'N/A')}")
        lines.append("")
    return "\n".join(lines)

def log_state(tag: str, state: "AgentState"):
    print(f"[{tag}] state snapshot:")
    for k, v in state.items():
        print(f"  {k}: {repr(v)}")

if __name__ == "__main__":
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin

    try:
        print("🔍 Submitting SmallWorld search...")
        hlid = smallworld_submit_search(test_smiles)
        print(f"✅ HLID received: {hlid}")

        print("\n📥 Viewing results...")
        results = smallworld_view_results(hlid, test_smiles)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        for i, item in enumerate(results, 1):
            print(item)
            #print(f"{i}. SMILES: {item['smiles']} | Similarity: {item['similarity']:.4f} | ID: {item['id']}")
    except Exception as e:
        print(f"❌ Error: {e}")