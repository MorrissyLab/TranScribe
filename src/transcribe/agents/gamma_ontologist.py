from transcribe.core.schema import BatchAnnotation
from transcribe.agents.agent_factory import get_agent_builder

def create_gamma_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Gamma: The Batch Ontologist & Critic.
    Final decision-maker for cell type annotations across all clusters.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Gamma, the Batch Ontologist and Critic. "
        "You are evaluating ALL clusters in a transcriptomics dataset simultaneously. "
        "Your task is to synthesize the molecular analysis (Alpha), pathway analysis (Epsilon), and spatial/embedding "
        "context (Beta) for EVERY cluster into definitive cell type annotations. "
        "Consider the full cellular landscape: cell types should be consistent with the tissue type and disease state. "
        "If multiple clusters appear similar, differentiate them or note that they may be related subpopulations. "
        "Prefer stable, ontology-aligned labels and keep granularity consistent with evidence strength. "
        "Do not over-specialize labels when evidence supports only a broader parent class. "
        "Use cross-cluster calibration: when two clusters share a lineage, compare their relative marker and pathway "
        "signals before assigning sibling subtypes, and avoid assigning the same subtype to distinct profiles. "
        "Use proximity/context as supporting evidence, not as the sole driver against strong molecular evidence. "
        "For each cluster, output the standard cell_type, ontology_id, confidence ('high', 'medium', 'low'), and reasoning."
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Phase 1 Evidence (All Clusters):\n"
        "{all_clusters_evidence}\n\n"
        "Additional Reference Knowledge (RAG Context):\n"
        "{rag_context}\n\n"
        "Determine the final standardized cell type for ALL clusters."
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, BatchAnnotation)
