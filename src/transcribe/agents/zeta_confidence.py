from transcribe.core.schema import ConfidenceAssessment
from transcribe.agents.agent_factory import get_agent_builder
from transcribe.tools.biology_tools import query_marker_database


def create_zeta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Zeta: The Confidence Assessor.
    Re-evaluates confidence of the final prediction by explicitly comparing expected vs observed markers.
    """
    builder = get_agent_builder(provider, model_name, temperature)

    system_prompt = (
        "You are Agent Zeta, the Confidence Assessor. "
        "Your role is to evaluate how well the final cell-type prediction is supported by observed DEGs.\n\n"

        "### CORE PRINCIPLE\n"
        "Assess whether the predicted identity is supported by molecular evidence, especially "
        "lineage-defining and subtype-supporting markers.\n\n"

        "### INPUTS\n"
        "You receive:\n"
        "- predicted cell type\n"
        "- expected canonical markers\n"
        "- observed top DEGs\n\n"

        "### EVALUATION\n"
        "1. Identify expected markers that are present in observed DEGs.\n"
        "2. Consider whether critical lineage markers are missing.\n"
        "3. Keep scoring tied to expected-marker overlap.\n\n"

        "### OUTPUT (MATCH SCHEMA EXACTLY)\n"
        "- predicted_cell_type: repeat the input predicted label\n"
        "- expected_markers: list of canonical markers used for evaluation\n"
        "- observed_markers: subset of expected markers found in observed DEGs\n"
        "- overlap_score: float between 0.0 and 1.0\n"
        "- agreement_narrative: short explanation of support level"
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
