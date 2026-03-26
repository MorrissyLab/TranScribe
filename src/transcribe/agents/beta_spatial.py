from transcribe.agents.agent_factory import get_agent_builder

def create_beta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Beta: The Spatial & Embedding Contextualizer.
    Evaluates neighborhood context using spatial neighbors or UMAP proximity.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Beta, the Spatial and Embedding Contextualizer. "
        "Your task is to evaluate the neighborhood context of a cell cluster "
        "and determine whether the proposed cell-type candidates are consistent "
        "with their surrounding clusters.\n\n"

        "### CORE PRINCIPLE\n"
        "Use neighborhood information as contextual evidence rather than primary "
        "identity evidence. Context should support or challenge candidate identities "
        "but should not override strong molecular signals.\n\n"

        "### DATA TYPES\n"
        "You may receive one of two types of context:\n"
        "- **Spatial neighborhood frequencies:** represent physical co-location of cell types "
        "within tissue.\n"
        "- **UMAP embedding proximity scores:** represent transcriptomic similarity "
        "between clusters in expression space.\n\n"

        "### INTERPRETATION GUIDELINES\n"
        "1. **Spatial Context:** Cells that frequently co-occur in tissue may represent "
        "interacting populations, shared microenvironments, or structural relationships.\n"
        "2. **Embedding Context:** Clusters that are close in UMAP space likely share "
        "similar transcriptional programs or lineage relationships.\n"
        "3. **Context Consistency Check:** Evaluate whether Alpha's candidate identities "
        "are compatible with the neighboring cluster types.\n"
        "4. **Caution Rule:** Neighborhood context alone should never redefine cell identity; "
        "it only provides supporting or contradictory evidence.\n\n"

        "### ANALYSIS APPROACH\n"
        "- First summarize the dominant neighboring cell types or clusters.\n"
        "- Then evaluate whether Alpha's candidate identities are consistent with "
        "this context.\n"
        "- If the context strongly contradicts Alpha's proposal, flag the inconsistency.\n\n"

        "### OUTPUT FORMAT\n"
        "contextual_adherence: [Plausible | Implausible | Inconclusive]\n"
        "context_summary: Brief description of the dominant neighboring clusters.\n"
        "critique: Concise explanation (max 3 sentences) of whether the proposed "
        "cell types fit the observed context."
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
    Summarizes UMAP proximity context for ALL clusters in one pass and returns
    a JSON object keyed by cluster id. It does not perform cell-type judgment.
    """
    builder = get_agent_builder(provider, model_name, temperature)

    system_prompt = (
        "You are Agent Beta, the Spatial and Embedding Contextualizer. "
        "You are evaluating ALL clusters in a single-cell dataset at once using UMAP proximity. "
        "UMAP proximity indicates transcriptomic relatedness (higher score = closer in embedding space). "
        "For each cluster, summarize only the neighborhood/proximity structure (nearest related clusters, "
        "isolation vs connectedness, and any notable proximity patterns). "
        "Do not make cell-type decisions and do not judge Alpha/Gamma annotations. "
        "Keep each summary concise (max 2 sentences). "
        "Return ONLY valid JSON with this shape: "
        "{{\"feedback_by_cluster\": {{\"<cluster_id>\": {{\"umap_context\": \"...\"}}}}}}. "
        "Do not wrap JSON in markdown."
    )

    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "All Cluster UMAP Context:\n"
        "{all_clusters_context}\n\n"
        "Provide the per-cluster UMAP context JSON."
    )

    return builder.build_string_chain(system_prompt, user_prompt)
