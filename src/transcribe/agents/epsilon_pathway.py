from transcribe.core.schema import PathwayAnalysis
from transcribe.agents.agent_factory import get_agent_builder

def create_epsilon_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Epsilon: The Intracellular Pathway Analyst.
    Analyzes Gene Ontology (GO) pathway enrichment data to summarize biological processes.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Epsilon, the Intracellular Pathway Analyst. "
        "Your task is to analyze Gene Ontology (GO) biological process enrichment results for a cell cluster. "
        "Summarize the key intracellular pathways active in this cluster and infer what cell type or "
        "functional state they suggest. "
        "You MUST evaluate the provided biological metadata (Organism, Tissue, Disease). "
        "Focus on pathways that define cell lineage, immune activation, or specific metabolic states. "
        "Consider the statistical significance (e.g. p-values or enrichment scores) if provided."
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Cluster ID: {cluster_id}\n"
        "Pathway Enrichment Data: {pathway_enrichment}\n\n"
        "Provide your pathway analysis summarizing the active biological processes."
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, PathwayAnalysis)
