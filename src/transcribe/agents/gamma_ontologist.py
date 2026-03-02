from transcribe.core.schema import FinalAnnotation
from transcribe.agents.factory import get_agent_builder

def create_gamma_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Gamma: The Ontologist & Critic.
    Final decision-maker for cell type annotation.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Gamma, the Ontologist and Critic. "
        "Your task is to synthesize the molecular analysis (Alpha) and spatial "
        "critique (Beta) into a single definitive cell type annotation. "
        "You MUST evaluate the provided biological metadata (Organism, Tissue, Disease)."
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Cluster ID: {cluster_id}\n"
        "Top DEGs: {top_degs}\n"
        "Alpha Candidates: {alpha_candidates}\n"
        "Beta Feedback: {beta_feedback}\n\n"
        "Additional Reference Knowledge (RAG Context):\n"
        "{rag_context}\n\n"
        "Determine the final standardized cell type."
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, FinalAnnotation)
