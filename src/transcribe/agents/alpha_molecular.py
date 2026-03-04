from transcribe.core.schema import CandidateList
from transcribe.agents.factory import get_agent_builder

def create_alpha_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Alpha: The Molecular Analyst.
    Analyzes purely transcriptomic signals to propose candidate cell types.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Alpha, the Molecular Analyst. "
        "Your task is to analyze transcriptomics to propose candidate cell types "
        "or functional states based exclusively on the given differential expression profile. "
        "You MUST evaluate the provided biological metadata (Organism, Tissue, Disease). "
        "For your confidence rating, you MUST output one of 'high', 'medium', or 'low'."
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Please analyze the following transcriptomics data:\n"
        "Cluster ID: {cluster_id}\n"
        "Top DEGs: {top_degs}\n"
        "Expression Profile: {expression_profile}\n"
        "Marker Overlap (Top 3 Genesets): {marker_overlap}\n"
        "Pathway Enrichment (Top 10 GO terms): {pathway_enrichment}\n"
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, CandidateList)
