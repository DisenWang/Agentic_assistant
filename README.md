# Agentic Assistant: Chemistry Research Chatbot

## Overview

This project was developed as part of the Agentic AI Scientist Take-Home Challenge.

---

## Features

### 1. **Molecular Structure Lookup (PubChem API)**
- Accepts chemical names such as "aspirin", "ibuprofen".
- Internally calls `get_smiles_from_pubchem()` from `utils.py`, which sends a GET request to the PubChem REST API.
- Extracts Canonical SMILES and Molecular Formula from the JSON response.
- SMILES and formula are stored in agent state for use by other nodes.

### 2. **Molecular Property Calculations (RDKit)**
- The `descriptor_node` uses `get_descriptors_from_smiles()` in `utils.py` to compute:
  - Molecular Weight (MW)
  - LogP
  - TPSA
- Accepts SMILES from memory or queries PubChem if not present.
- Results are cached in state and selectively returned based on user input.

### 3. **Protein Structure Metadata (PDB)**
- Accepts valid PDB IDs from `REAL-Database-22Q1` as user input.
- Calls `get_pdb_metadata()` in `utils.py` to fetch XML from RCSB PDB REST endpoint.
- Parses organism, expression system, resolution, deposition and release dates.
- Presents the result as structured metadata.

### 4. **Chemical Similarity Search (SmallWorld API)**
- Accepts SMILES as input (auto-fetched from memory if missing).
- Calls `smallworld_submit_search()` and `smallworld_view_results()` in `utils.py`.
- Submits similarity search and parses TSV results.
- Re-ranks results using RDKit-based Tanimoto similarity.
- Returns top 5 matches with SMILES and similarity scores.

### 5. **Web Search Integration**
- Triggered by general science or chemistry-related queries.
- Uses SerpAPI to query Google Search.
- Extracts snippets and links from search results and summarizes for the user.

### 6. **Patent Lookup**
- Triggered by queries containing keywords like “patent”, “filed”, “approved” and a target (e.g., GLP-1).
- Uses USPTO API or SerpAPI to locate patents related to the compound or target.
- Parses and returns patent title, inventors, publication date.

---

## Local Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/DisenWang/Agentic_assistant.git
cd Agentic_assistant
```

### 2. Create Environment
Use `conda` for RDKit compatibility:
```bash
conda create -n chemagent python=3.9
conda activate chemagent
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

### 3. Set Environment Variables
Make sure to setup SERPAPI_KEY and OPENAI_API_KEY by creating a `.env` file or export manually:
```bash
export OPENAI_API_KEY=your_openai_key
export SERPAPI_KEY=your_serpapi_key
```

### 4. Run Locally
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

---

### Demo Video  
[Watch the demonstration video](https://youtu.be/rNmm7dYg5FY)

---

## Architecture

- **Frontend:** Streamlit for interactive UI
- **LLM Backend:** OpenAI API for intent classification and fallback QA
- **Agentic Flow:** LangGraph for orchestrating multi-step logic
- **Memory:** Conversation history retained across 4 previous user/assistant turns

---

## API Integrations

| API         | Purpose                                 | Error Handling        |
|-------------|-----------------------------------------|------------------------|
| PubChem     | Retrieve structure/SMILES               | Graceful 404 handling  |
| RDKit       | Descriptor calculations                 | Internal validation    |
| PDB         | Protein metadata query                  | URL fallback & timeout |
| SmallWorld  | Similar molecule search                 | Fallback + retry       |
| SerpAPI/USPTO | Patent lookup                         | API key + structured response parsing |

