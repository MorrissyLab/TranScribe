from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from typing_extensions import TypedDict

class CandidateAnnotation(BaseModel):
    """A single candidate cell type annotation."""
    cell_type: str = Field(description="Proposed cell type name.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of this candidate: 'high', 'medium', or 'low'.")
    reasoning: str = Field(description="Brief reasoning for this annotation.")

class CandidateList(BaseModel):
    """A list of proposed candidate cell types."""
    candidates: List[CandidateAnnotation] = Field(description="List of candidates sorted by confidence.")

class FinalAnnotation(BaseModel):
    """The final finalized annotation by Agent Gamma."""
    cluster_id: str = Field(description="The cluster being annotated.")
    cell_type: str = Field(description="Standardized Cell Ontology type.")
    ontology_id: str = Field(default="", description="Optional Cell Ontology ID (e.g., CL:0000000).")
    confidence: Literal["high", "medium", "low"] = Field(description="Final categorical confidence level: 'high', 'medium', or 'low'.")
    reasoning_chain: str = Field(description="Full reasoning chain explaining the decision.")

class EvaluationMatch(BaseModel):
    """Result of Agent Delta comparing true vs predicted labels."""
    is_match: bool = Field(description="True if labels are biologically equivalent.")
    explanation: str = Field(description="Explanation of why they match or differ.")

class ClusterEvaluation(BaseModel):
    """Evaluation result for a single cluster within a batch."""
    cluster_id: str
    predicted_label: str
    true_label: str
    is_match: bool
    explanation: str

class BatchEvaluation(BaseModel):
    """The result of Delta evaluating all clusters in a single prompt."""
    evaluations: List[ClusterEvaluation]

class PathwayAnalysis(BaseModel):
    """Result of Agent Epsilon analyzing GO enrichment pathways."""
    top_pathways: List[str] = Field(description="List of the most prominent biological pathways.")
    biological_summary: str = Field(description="Narrative summary of the active pathways and their cellular implications.")
    suggested_cell_states: List[str] = Field(description="Cell types or functional states suggested by pathway evidence.")

class ConfidenceAssessment(BaseModel):
    """Result of Agent Zeta evaluating predicted vs expected marker presence."""
    predicted_cell_type: str = Field(description="The cell type predicted by Gamma.")
    expected_markers: List[str] = Field(description="Canonical genes expected for this cell type.")
    observed_markers: List[str] = Field(description="Expected genes that were actually found in the input DEGs.")
    overlap_score: float = Field(description="Fraction of expected markers present (0.0 to 1.0).")
    agreement_narrative: str = Field(description="Brief explanation of the agreement score.")

class BatchAnnotation(BaseModel):
    """Result of batch Gamma processing multiple clusters simultaneously."""
    annotations: List[FinalAnnotation] = Field(description="Final annotations for all provided clusters.")

class CellTypeGroup(BaseModel):
    """A hierarchical group produced by Eta."""
    group_name: str = Field(description="Name of the group (e.g. 'T Cells').")
    parent_group: str = Field(description="Parent group category (e.g. 'Immune').")
    member_clusters: List[str] = Field(description="Cluster IDs belonging to this group.")
    description: str = Field(description="Brief description of the cell types in this group.")

class HierarchicalSummary(BaseModel):
    """Result of Agent Eta organizing cell types into a hierarchy."""
    groups: List[CellTypeGroup] = Field(description="Hierarchical grouping of the identified cell types.")
    narrative_summary: str = Field(description="A paragraph summarizing the cellular composition of the dataset.")

class AgentState(TypedDict):
    """The overall state passed through the LangGraph workflow."""
    cluster_id: str
    metadata: Dict[str, str]  # tissue, organism, disease
    top_degs: List[str]
    expression_profile: Optional[Dict[str, float]]
    spatial_neighbor_frequencies: Optional[Dict[str, float]]
    marker_overlap: Optional[Dict[str, float]]
    pathway_enrichment: Optional[Dict[str, float]]
    alpha_candidates: Optional[CandidateList]
    pathway_analysis: Optional[PathwayAnalysis]
    beta_feedback: Optional[str]
    final_annotation: Optional[FinalAnnotation]
    confidence_assessment: Optional[ConfidenceAssessment]
    messages: List[Any]  # Used to track the "debate" dialog history
    errors: List[str]
