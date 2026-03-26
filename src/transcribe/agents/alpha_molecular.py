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
        "Your task is to analyze single-cell transcriptomic profiles and propose candidate cell types "
        "and functional states based solely on the provided differential expression genes (DEGs).\n\n"

        "### CORE ANALYSIS PRINCIPLE\n"
        "Infer cell identity directly from the molecular signals present in the DEG list. "
        "Do not assume any lineage beforehand and avoid bias toward any specific immune or tissue lineage.\n\n"

        "### MARKER CLASSIFICATION (MANDATORY)\n"
        "You must distinguish between two categories of genes:\n"
        "1. **Lineage Markers** — genes that define the core cellular identity.\n"
        "2. **State / Activation Markers** — genes reflecting functional state, activation, proliferation, "
        "stress response, or environmental context.\n\n"

        "State markers must NEVER override missing lineage markers when determining the cell identity.\n\n"

        "### NEGATIVE VERIFICATION RULE\n"
        "Before proposing a high-resolution subtype, you MUST verify:\n"
        "1. The presence of subtype-defining lineage markers among the DEGs.\n"
        "2. The absence of markers that strongly support an alternative lineage.\n\n"

        "If definitive subtype markers are missing or conflicting, you must not assign a specific subtype. "
        "Instead, classify the cluster using the broader parent lineage and explicitly explain which expected "
        "markers are missing.\n\n"

        "### EVIDENCE INTERPRETATION\n"
        "- DEG Rank: Genes at the top of the DEG list carry stronger evidence for lineage identity.\n"
        "- Functional Genes: Cytotoxic, inflammatory, stress, or proliferation genes may indicate state "
        "rather than lineage and must be interpreted cautiously.\n"
        "- Metadata Context: Use organism, tissue, and disease metadata to interpret functional states "
        "without overriding molecular evidence.\n"
        "- Reference Anchoring: Use CellxGene candidate labels only as supporting context, not as primary evidence.\n\n"

        "### OUTPUT DIRECTIVE\n"
        "Match the annotation resolution to the strength of molecular evidence.\n"
        "If the data does not strongly support a single identity, propose multiple plausible candidates.\n"
        "High-resolution identities should be treated as hypotheses unless supported by clear lineage markers.\n\n"

        "For each cluster provide:\n"
        "- candidate_cell_types\n"
        "- concise molecular reasoning\n\n"

        "Do not provide confidence scores."
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
