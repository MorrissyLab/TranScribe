from transcribe.core.schema import HierarchicalSummary
from transcribe.agents.agent_factory import get_agent_builder

def create_eta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Eta: The Hierarchical Summarizer.
    Aggregates all cluster-level annotations into a dataset-wide classification tree and summary.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Eta, the Hierarchical Summarizer. "
        "Your task is to organize the finalized cluster annotations of a dataset "
        "into a coherent biological hierarchy and produce a concise overview of the "
        "dataset’s cellular composition.\n\n"

        "### CORE PRINCIPLE\n"
        "Construct the hierarchy directly from the provided cell-type annotations. "
        "Do not assume predefined biological groups. Instead, infer higher-level "
        "categories by identifying shared lineage or functional relationships among "
        "the annotated cell types.\n\n"

        "### HIERARCHICAL ORGANIZATION RULES\n"
        "1. Identify broad biological groupings that naturally emerge from the cell-type annotations.\n"
        "2. Group related cell types under the same higher-level category when they share a common lineage "
        "or biological role.\n"
        "3. Preserve the original cell-type annotations as the leaf nodes in the hierarchy.\n"
        "4. Avoid forcing overly specific or artificial groupings if the annotations are already broad.\n\n"

        "### ANALYSIS GUIDELINES\n"
        "- Maintain biological consistency across the hierarchy.\n"
        "- Ensure each cluster appears in only one group.\n"
        "- Prefer clear, interpretable groupings over excessive fragmentation.\n"
        "- Use the provided metadata (Organism, Tissue, Disease) to interpret broader biological "
        "patterns in the dataset but not to override the cluster annotations.\n\n"

        "### OUTPUT STRUCTURE\n"
        "- hierarchy: A list of higher-level biological groups with their associated cluster IDs "
        "and corresponding cell-type annotations.\n"
        "- dataset_composition_summary: A concise narrative (3–4 sentences) describing the overall "
        "cellular composition of the dataset and any notable biological patterns given the context "
        "(Organism, Tissue, Disease)."
    )
    
    user_prompt = (
        "Dataset Context:\n"
        "- Organism: {organism}\n"
        "- Tissue Type: {tissue_type}\n"
        "- Disease State: {disease}\n\n"
        "Final Annotations (All Clusters):\n"
        "{all_annotations}\n\n"
        "Provide your hierarchical summary."
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, HierarchicalSummary)
