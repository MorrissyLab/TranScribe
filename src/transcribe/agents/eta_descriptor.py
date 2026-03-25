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
        "Your task is to take a complete list of cell-type annotations for all clusters "
        "in a dataset and organize them into a logical biological hierarchy. "
        "Group related cell types into higher-level categories (e.g., 'T Cells', 'Myeloid Cells', "
        "'Stromal Cells'). For each group, list the cluster IDs that belong to it. "
        "Finally, write a brief narrative summary of the cellular composition of the entire dataset, "
        "highlighting any biologically interesting or overarching patterns given the "
        "provided contextual metadata (Organism, Tissue, Disease)."
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
