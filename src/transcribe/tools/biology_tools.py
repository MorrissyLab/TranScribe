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

@tool
def query_marker_database(cell_type: str) -> List[str]:
    """
    Input: A proposed cell type name.
    Output: A list of canonical marker genes for that cell type according to standard literature.
    """
    # Placeholder: Mock DB lookup. Can be extended with CellTypist or similar database queries.
    mock_db = {
        "CD4+ T cell": ["CD4", "IL7R", "CD3E"],
        "CD8+ T cell": ["CD8A", "CD8B", "CD3E", "GZMB"],
        "B cell": ["CD79A", "MS4A1", "CD19"],
        "Macrophage": ["CD14", "FCGR3A", "CD68"]
    }
    return mock_db.get(cell_type, ["Unknown", "markers"])

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
