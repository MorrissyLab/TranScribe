from typing import Any
from transcribe.agents.factory import get_agent_builder
from transcribe.core.schema import EvaluationMatch

def create_delta_agent(provider: str = "gemini", model_name: str = "gemma-3-4b-it"):
    """
    Agent Delta: The Evaluator.
    Responsibility: Compare the ground truth label with the predicted label 
    and determine if they are biologically equivalent (synonyms).
    """
    system_prompt = """
    You are Agent Delta, a specialist in single-cell ontology and biological nomenclature.
    Your task is to evaluate the accuracy of cell type predictions.
    
    You will be given:
    1. A 'Ground Truth' label (the known correct cell type).
    2. A 'Predicted' label (the annotation produced by an AI system).
    
    You must determine if the 'Predicted' label is a correct biological match for the 'Ground Truth'.
    - Use 'is_match: true' if the labels are synonyms, represent the same lineage at reasonable granularity, 
      or if the prediction is a correct sub-type or parent-type of the ground truth (e.g., 'Monocyte' and 'CD14+ Monocyte' are matches).
    - Use 'is_match: false' if they represent fundamentally different lineages (e.g., 'B cell' and 'T cell').
    
    Provide a brief explanation for your decision.
    """
    
    user_prompt = """
    Evaluate the following pair:
    Ground Truth: {true_label}
    Predicted: {predicted_label}
    
    Respond in the requested structured format.
    """
    
    builder = get_agent_builder(provider, model_name)
    return builder.build_structured_chain(system_prompt, user_prompt, EvaluationMatch)
