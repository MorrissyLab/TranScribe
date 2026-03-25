from transcribe.agents.agent_factory import get_agent_builder

def create_beta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Beta: The Spatial & Embedding Contextualizer.
    Evaluates neighborhood context using spatial neighbors or UMAP proximity.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Beta, the Spatial and Embedding Contextualizer. "
        "Your task is to evaluate the neighborhood context of a cell cluster. "
        "You may receive EITHER spatial neighborhood frequencies (from spatial transcriptomics) "
        "OR UMAP embedding proximity scores (from single-cell RNA-seq). "
        "For spatial data, these frequencies represent physical co-location of cell types. "
        "For UMAP data, proximity scores represent transcriptomic similarity between clusters "
        "(higher score = more similar embedding, potentially related lineage or function). "
        "Determine if Alpha's proposed candidates are biologically plausible given the "
        "neighboring cluster context.\n\n"
        "Format Shape:\n"
        "Contextual Adherence: [Plausible/Implausible]\n"
        "Critique: [Max 3 sentences]"
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Cluster ID: {cluster_id}\n"
        "Alpha Candidates: {alpha_candidates}\n"
        "Neighborhood Context: {spatial_neighbors}\n\n"
        "Provide your contextual feedback report:"
    )
    
    # We add a simple string chain helper to the builder
    return builder.build_string_chain(system_prompt, user_prompt)


def create_beta_batch_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Batch Beta agent for single-cell mode.
    Reviews UMAP proximity context for ALL clusters in one pass and returns
    a JSON object keyed by cluster id.
    """
    builder = get_agent_builder(provider, model_name, temperature)

    system_prompt = (
        "You are Agent Beta, the Spatial and Embedding Contextualizer. "
        "You are evaluating ALL clusters in a single-cell dataset at once using UMAP proximity. "
        "UMAP proximity indicates transcriptomic relatedness (higher score = closer in embedding space). "
        "For each cluster, judge whether Alpha's candidate is plausible in the context of nearby clusters. "
        "Return ONLY JSON in this shape: "
        "{{\"<cluster_id>\": {{\"contextual_adherence\": \"Plausible|Implausible\", \"critique\": \"...\"}}}}. "
        "Keep each critique concise (max 2 sentences)."
    )

    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "All Cluster UMAP Context:\n"
        "{all_clusters_context}\n\n"
        "Provide the per-cluster contextual feedback JSON."
    )

    return builder.build_string_chain(system_prompt, user_prompt)
