from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict

class CandidateAnnotation(BaseModel):
    """A single candidate cell type annotation."""
    cell_type: str = Field(description="Proposed cell type name.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief reasoning for this annotation.")

class CandidateList(BaseModel):
    """A list of proposed candidate cell types."""
    candidates: List[CandidateAnnotation] = Field(description="List of candidates sorted by confidence.")

class FinalAnnotation(BaseModel):
    """The final finalized annotation by Agent Gamma."""
    cluster_id: str = Field(description="The cluster being annotated.")
    cell_type: str = Field(description="Standardized Cell Ontology type.")
    ontology_id: str = Field(default="", description="Optional Cell Ontology ID (e.g., CL:0000000).")
    confidence: float = Field(description="Final confidence score.")
    reasoning_chain: str = Field(description="Full reasoning chain explaining the decision.")

class EvaluationMatch(BaseModel):
    """Result of Agent Delta comparing true vs predicted labels."""
    is_match: bool = Field(description="True if labels are biologically equivalent.")
    explanation: str = Field(description="Explanation of why they match or differ.")

class AgentState(TypedDict):
    """The overall state passed through the LangGraph workflow."""
    cluster_id: str
    metadata: Dict[str, str]  # tissue, organism, disease
    top_degs: List[str]
    expression_profile: Optional[Dict[str, float]]
    spatial_neighbor_frequencies: Optional[Dict[str, float]]
    alpha_candidates: Optional[CandidateList]
    beta_feedback: Optional[str]
    final_annotation: Optional[FinalAnnotation]
    messages: List[Any]  # Used to track the "debate" dialog history
    errors: List[str]
