from transcribe.agents.agent_factory import get_agent_builder
from transcribe.core.schema import BatchEvaluation

def create_delta_agent(provider: str = "gemini", model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    """
    Agent Delta: The Evaluator (Batched).
    Responsibility: Compare the ground truth label with the predicted label 
    for ALL clusters in a single prompt and determine if they match.
    """
    system_prompt = """
    You are Agent Delta, a specialist in single-cell ontology and biological nomenclature.
    Your task is to evaluate the accuracy of multiple cell type predictions at once.
    
    For each cluster, you will be given:
    1. A 'Ground Truth' label (if available).
    2. A 'Predicted' label (produced by our multi-agent system).
    3. 'Input Context' (Top marker genes).
    
    You must determine if each 'Predicted' label is a correct biological match for its 'Ground Truth'.
    If Ground Truth is 'Unknown', evaluate if the 'Predicted' label is consistent with the 'Input Context'.
    
    Match Criteria:
    - is_match: true if labels are synonyms, lexical variants, parent/child types, maturation-state variants,
      or otherwise biologically compatible with the same lineage and marker context.
    - is_match: false if lineages are distinct or discordant with markers.
    
    Provide a brief explanation for each cluster. Respond in the requested BatchEvaluation format.
    """
    
    user_prompt = """
    Evaluate the following clusters:
    {eval_input}
    
    Compare Predicted vs Ground Truth/Input for each.
    """
    
    builder = get_agent_builder(provider, model_name, temperature)
    return builder.build_structured_chain(system_prompt, user_prompt, BatchEvaluation)
