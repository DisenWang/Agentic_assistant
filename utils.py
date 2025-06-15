import os
import time
import json
import asyncio
import requests
import aiohttp
from typing import TypedDict, Optional
from langchain_core.messages.base import BaseMessage
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from serpapi import GoogleSearch


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

async def get_smiles_from_pubchem(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,MolecularFormula/JSON"
    print(f"[PubChem][GET] {url}")
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"[PubChem][Status] {response.status}")
            if response.status == 200:
                try:
                    data = await response.json()
                    props = data["PropertyTable"]["Properties"][0]
                    return props["CanonicalSMILES"], props["MolecularFormula"]
                except Exception as e:
                    raise RuntimeError(f"[PubChem] JSON Parsing Error: {e}")
            else:
                raise RuntimeError(f"[PubChem] Failed with status {response.status}: {await response.text()}")


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



async def smallworld_submit_search(smiles: str, db: str = db, dist: int = 4, top: int = 5):
    url = "https://sw.docking.org/search/submit"
    params = {
        "smi": [smiles],
        "db": db,
        "dist": str(dist),
        "top": str(top)
    }
    print(f"[SmallWorld Submit] Submitting search...")
    print(f"[SmallWorld][Submit][GET] {url} with {params}")
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            print(f"[SmallWorld][Submit][Status] {resp.status}")
            if resp.status != 200:
                raise Exception(f"Submit error {resp.status}: {await resp.text()}")

            text = await resp.text()
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if "hlid" in data and data["status"] == "END":
                        return data["hlid"]
    raise Exception("HLID not found in response")


async def smallworld_view_results(hlid: int, state_smiles: str):
    url = "https://sw.docking.org/search/view"
    params = {
        "hlid": hlid,
        "fmt": "tsv",
        "scores": "graph"
    }
    print(f"[SmallWorld View] Fetching results for HLID {hlid} ...")
    print(f"[SmallWorld][View][GET] {url} with {params}")
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            print(f"[SmallWorld][View][Status] {resp.status}")
            if resp.status != 200:
                raise Exception(f"View error {resp.status}: {await resp.text()}")

            lines = (await resp.text()).strip().splitlines()
            print(f"[SmallWorld View] Raw TSV lines: {lines}")
            results = []
            for line in lines:
                if line.strip().lower().startswith("smiles"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                smiles_and_id = parts[0]
                try:
                    smiles, zid = smiles_and_id.rsplit(" ", 1)
                except ValueError:
                    smiles, zid = smiles_and_id, "N/A"
                try:
                    query_mol = Chem.MolFromSmiles(state_smiles)
                    result_mol = Chem.MolFromSmiles(smiles)
                    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
                    result_fp = AllChem.GetMorganFingerprintAsBitVect(result_mol, 2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(query_fp, result_fp)
                except Exception as e:
                    print(f"[Similarity Error] Failed for {smiles}: {e}")
                    similarity = 0.0
                results.append({"id": zid, "smiles": smiles, "similarity": similarity})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:5]


def query_google_patents_via_serpapi(query: str, serpapi_key: str, num_results: int = 10):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": serpapi_key,
        "num": num_results
    }
    for attempt in range(3):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[SerpAPI] Attempt {attempt+1} failed with {response.status_code}")
            time.sleep(1)
    return f"Error: {response.status_code} – {response.text}"


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


def calculate_molar_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


def calculate_tpsa(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.TPSA(mol)


def calculate_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol)


def draw_molecule_image(smiles: str):
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)


def add_messages(memory: list[BaseMessage], *new_messages: BaseMessage) -> list[BaseMessage]:
    return memory + list(new_messages)

# if __name__ == "__main__":
#     test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin

#     try:
#         print("[SmallWorld Submit] Submitting search...")
#         hlid = smallworld_submit_search(test_smiles)
#         print(f"[SmallWorld Submit] HLID received: {hlid}")

#         print("\n[SmallWorld View] Viewing results...")
#         results = smallworld_view_results(hlid, test_smiles)
#         results.sort(key=lambda x: x['similarity'], reverse=True)
#         for i, item in enumerate(results, 1):
#             print(item)
#             # print(f"{i}. SMILES: {item['smiles']} | Similarity: {item['similarity']:.4f} | ID: {item['id']}")
#     except Exception as e:
#         print(f"Error: {e}")