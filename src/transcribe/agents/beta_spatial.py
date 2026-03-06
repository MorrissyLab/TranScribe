from transcribe.agents.agent_factory import get_agent_builder

def create_beta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Beta: The Spatial Contextualizer.
    Evaluates spatial adherence based on neighborhood frequencies.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Beta, the Spatial Contextualizer. "
        "Your task is to evaluate the spatial neighborhood (nichecard) of a cell cluster. "
        "Determine if Alpha's proposed candidates are biologically plausible in this context.\n\n"
        "Format Shape:\n"
        "Spatial Adherence: [Plausible/Implausible]\n"
        "Critique: [Max 3 sentences]"
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Cluster ID: {cluster_id}\n"
        "Alpha Candidates: {alpha_candidates}\n"
        "Spatial Neighbors Frequencies: {spatial_neighbors}\n\n"
        "Provide your spatial feedback report:"
    )
    
    # We add a simple string chain helper to the builder
    return builder.build_string_chain(system_prompt, user_prompt)
