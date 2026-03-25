from langchain_core.tools import tool
from typing import List, Dict, Any

@tool
def gsea_tool(genes: List[str]) -> Dict[str, float]:
    """
    Input: A list of gene symbols (e.g., ['CD4', 'CD8A']).
    Output: A dictionary of pathways or GO-terms with their corresponding enrichment scores.
    Uses an external API or mocked locally for evaluation purposes.
    """
    # Placeholder: currently returns mock pathways. 
    # In production, this can connect to Enrichr API, GSEApy, or local databases.
    return {
        "T cell activation": 0.95,
        "immune response": 0.88,
        "lymphocyte differentiation": 0.75
    }

from transcribe.agents.agent_factory import get_agent_builder
import json

@tool
def query_marker_database(cell_type: str, organism: str = "Human", tissue: str = "PBMC", provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", gamma_reasoning: str = "") -> List[str]:
    """
    Input: A proposed cell type name and context (organism/tissue).
    Output: A list of canonical marker genes.
    """
    # Robust fallback mapping
    fallback_db = {
        "CD4 T": ["CD4", "IL7R", "CD3E", "CD3D", "CD2"],
        "CD8 T": ["CD8A", "CD8B", "CD3E", "GZMB", "PRF1"],
        "B cell": ["CD79A", "MS4A1", "CD19", "CD79B"],
        "NK cell": ["NKG7", "GNLY", "FCGR3A", "NCAM1", "KLRD1"],
        "Monocyte": ["CD14", "LYZ", "S100A9", "S100A8", "FCN1"],
        "Classical Monocyte": ["CD14", "S100A8", "S100A9", "LYZ"],
        "Dendritic cell": ["FCER1A", "CST3", "HLA-DQA1", "CD1C", "CLEC10A"],
        "Megakaryocyte": ["PPBP", "PF4", "GP9", "GP1BA"],
        "Platelet": ["PPBP", "PF4", "GNG11"],
        "Erythrocyte": ["HBA1", "HBA2", "HBB"]
    }
    
    ct_lower = cell_type.lower()
    # Priority matching: exact or strong substring
    for key, markers in fallback_db.items():
        if key.lower() in ct_lower or ct_lower in key.lower():
             return markers
            
    # LLM-based query for specific subtypes (e.g. "CD56dim NK cell", "Naive B cell")
    builder = get_agent_builder(provider, model_name)
    reasoning_hint = f" Reasoning context from Gamma: {gamma_reasoning}" if gamma_reasoning else ""
    sys_msg = f"You are a biology database. Return ONLY a JSON list of 5-8 canonical gene symbols for the cell type '{cell_type}' in {organism} {tissue}.{reasoning_hint} No prose."
    try:
         from langchain_core.messages import HumanMessage
         res = builder.llm.invoke([HumanMessage(content=sys_msg)])
         clean_str = res.content.strip()
         if "```json" in clean_str:
              clean_str = clean_str.split("```json")[-1].split("```")[0]
         elif "```" in clean_str:
              clean_str = clean_str.split("```")[-1].split("```")[0]
         
         if "[" in clean_str and "]" in clean_str:
              start = clean_str.find("[")
              end = clean_str.rfind("]") + 1
              clean_str = clean_str[start:end]
              
         markers = json.loads(clean_str.strip())
         if isinstance(markers, list) and len(markers) > 0:
              return [str(m).upper() for m in markers]
    except Exception:
         pass
         
    # Final safety net for immune clusters (more robust matching)
    if "t cell" in ct_lower or "t-cell" in ct_lower: return fallback_db["CD4 T"]
    if "b cell" in ct_lower: return fallback_db["B cell"]
    if "monocyte" in ct_lower: return fallback_db["Monocyte"]
    if "nk cell" in ct_lower or "nk-cell" in ct_lower: return fallback_db["NK cell"]
    if "dendritic" in ct_lower: return fallback_db["Dendritic cell"]
    if "megakaryocyte" in ct_lower: return fallback_db["Megakaryocyte"]
    
    return ["CD3E", "CD19", "CD14", "NKG7"] # Absolute fallback for immune cells

@tool
def check_cell_ontology(cell_type: str) -> Dict[str, Any]:
    """
    Input: A proposed cell type string.
    Output: The standardized Cell Ontology (CL) exact match or closest known parent, with its definition.
    """
    # Mock lookup
    return {
        "standard_name": cell_type,
        "ontology_id": "CL:0000000", 
        "valid": True
    }
