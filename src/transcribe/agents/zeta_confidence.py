from transcribe.core.schema import ConfidenceAssessment
from transcribe.agents.agent_factory import get_agent_builder
from transcribe.tools.biology_tools import query_marker_database

def create_zeta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Zeta: The Confidence Assessor.
    Re-evaluates the confidence of the final prediction by explicitly comparing expected vs observed markers.
    """
    builder = get_agent_builder(provider, model_name, temperature)
    
    system_prompt = (
        "You are Agent Zeta, the Confidence Assessor. "
        "Your task is to review the final cell type prediction made by Agent Gamma against the raw data. "
        "You will be given the predicted cell type, its expected canonical marker genes, and the "
        "observed top differentially expressed genes (DEGs) for the cluster. "
        "Determine which expected markers were actually observed, compute a rough overlap score "
        "(0.0 to 1.0, where 1.0 means all expected key markers are present), and provide a brief "
        "narrative explaining your confidence assessment. "
        "If expected markers are missing but similar functional genes are present, you may adjust "
        "the score favorably. If key lineage markers are absent, the score should be low."
    )
    
    user_prompt = (
        "Cluster ID: {cluster_id}\n"
        "Gamma Prediction: {predicted_cell_type}\n"
        "Gamma Reasoning: {gamma_reasoning}\n"
        "Expected Canonical Markers: {expected_markers}\n"
        "Observed Top DEGs: {observed_degs}\n\n"
        "Provide your confidence assessment."
    )
    
    return builder.build_structured_chain(system_prompt, user_prompt, ConfidenceAssessment)
