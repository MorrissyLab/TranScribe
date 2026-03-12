from transcribe.core.schema import CandidateList
from transcribe.agents.agent_factory import get_agent_builder

def create_alpha_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Alpha: The Molecular Analyst.
    Analyzes purely transcriptomic signals to propose candidate cell types.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Alpha, the Molecular Analyst. "
        "Your task is to analyze transcriptomics to propose candidate cell types "
        "or functional states based on the provided differential expression profile. "
        "IMPORTANT: The 'Top DEGs' list is RANKED by statistical significance and fold-change; "
        "you MUST prioritize genes at the top of the list as the most definitive markers. "
        "You MUST evaluate the provided biological metadata (Organism, Tissue, Disease). "
        "If the Disease State is 'Cancer', you MUST consider the tumor microenvironment (TME) context, "
        "including immune infiltration, stromal components, and tumor-specific states. "
        "If the Disease State is 'Normal', you MUST prioritize identifying standard physiological cell types and resident populations characteristic of healthy tissue. "
        "Always aim for fine-grained cell subtypes (e.g., 'CD8+ Effector T-cell' instead of just 'T-cell') "
        "whenever the marker evidence supports it. "
        "If there is any ambiguity, you MUST propose multiple candidates in your list "
        "so that Agent Gamma can make the final decision based on all available evidence. "
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
        "{data_payload}"
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, CandidateList)
