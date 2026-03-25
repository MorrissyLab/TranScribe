from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from transcribe.core.schema import AgentState
from transcribe.agents.alpha_molecular import create_alpha_agent
from transcribe.agents.beta_spatial import create_beta_agent
from transcribe.agents.epsilon_pathway import create_epsilon_agent
from transcribe.tools.rag.retriever import retrieve_rag_context
from transcribe.tools.cellxgene_annotator import CellxGeneAnnotator
from transcribe.config import logger, DEFAULT_MODEL_NAME

def build_workflow(provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, modality: str = "single-cell", use_rag: bool = False, rag_index: str = ""):
    """
    Constructs the LangGraph workflow.
    - single-cell: Alpha -> Epsilon (Batch Beta is handled later in inference_engine)
    - spatial: Alpha -> Epsilon -> Beta
    """
    alpha = create_alpha_agent(provider=provider, model_name=model_name)
    beta = create_beta_agent(provider=provider, model_name=model_name)
    epsilon = create_epsilon_agent(provider=provider, model_name=model_name)

    workflow = StateGraph(AgentState)

    # Shared CellxGene annotator instance (lazy-initialized)
    cellxgene_annotator = None

    def run_alpha(state: AgentState):
        nonlocal cellxgene_annotator
        meta = state.get("metadata", {})
        
        # Build dynamic data payload to omit empty fields
        data_parts = []
        mo = state.get("marker_overlap")
        if mo and str(mo).lower() != "none available":
            data_parts.append(f"Marker Overlap (Top 3 Genesets): {mo}")
        
        # Query CellxGene WMG for population-level evidence
        cxg_str = "None"
        try:
            organism = meta.get("organism", "Human")
            tissue = meta.get("tissue_type", "Unknown")
            if cellxgene_annotator is None:
                cellxgene_annotator = CellxGeneAnnotator(organism=organism)
            cxg_result = cellxgene_annotator.query(state["top_degs"][:20], tissue=tissue)
            if cxg_result and cxg_result.get("candidates"):
                cxg_str = ", ".join([f"{name} ({score:.3f})" for name, score in cxg_result["candidates"][:5]])
                data_parts.append(f"CellxGene WMG Candidates: {cxg_str}")
        except Exception as e:
            logger.warning(f"CellxGene query failed for cluster {state['cluster_id']}: {e}")
            
        data_payload = "\n".join(data_parts)

        logger.info(f"[Agent Call] Alpha | cluster={state['cluster_id']}")
        result = alpha.invoke({
            "organism": meta.get("organism", "Unknown"),
            "tissue_type": meta.get("tissue_type", "Unknown"),
            "disease": meta.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "top_degs": state["top_degs"],
            "data_payload": data_payload
        })
        messages = state.get("messages", [])
        
        # Enhanced Trace: Show the exact evidence Alpha saw
        alpha_input = f"Top DEGs: {state['top_degs'][:20]}"
        if data_payload:
            alpha_input += f"\n{data_payload}"
        messages.append({
            "role": "Alpha Input Evidence", 
            "content": alpha_input
        })
        messages.append({
            "role": "Alpha Annotation", 
            "content": result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)
        })
        return {"alpha_candidates": result, "messages": messages}

    def run_epsilon(state: AgentState):
        from transcribe.tools.biology_tools import gsea_tool
        meta = state.get("metadata", {})
        
        # Call GSEA Tool programmatically (programmatic calling as requested)
        try:
            pathway_results = gsea_tool.func(state["top_degs"][:50])
            pe = str(pathway_results)
        except Exception as e:
            logger.warning(f"GSEA Tool failed for cluster {state['cluster_id']}: {e}")
            pe = state.get("pathway_enrichment") or "None available"
        
        if not pe or str(pe).lower() == "none available":
            return {"pathway_analysis": None}

        logger.info(f"[Agent Call] Epsilon | cluster={state['cluster_id']}")
        result = epsilon.invoke({
            "organism": meta.get("organism", "Unknown"),
            "tissue_type": meta.get("tissue_type", "Unknown"),
            "disease": meta.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "pathway_enrichment": pe
        })
        messages = state.get("messages", [])
        messages.append({
            "role": "Epsilon Input",
            "content": f"Genes: {state['top_degs'][:50]}"
        })
        messages.append({
            "role": "Epsilon Output",
            "content": (
                f"GSEA Scores: {pe}\n\n"
                f"Pathway Analysis: "
                f"{result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)}"
            )
        })
        return {"pathway_analysis": result, "messages": messages}

    def run_beta(state: AgentState):
        meta = state.get("metadata", {})
        logger.info(f"[Agent Call] Beta | cluster={state['cluster_id']}")
        result = beta.invoke({
            "organism": meta.get("organism", "Unknown"),
            "tissue_type": meta.get("tissue_type", "Unknown"),
            "disease": meta.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "alpha_candidates": state.get("alpha_candidates"),
            "spatial_neighbors": state.get("spatial_neighbor_frequencies")
        })
        messages = state.get("messages", [])
        messages.append({
            "role": "Beta Spatial Critique", 
            "content": str(result)
        })
        return {"beta_feedback": result, "messages": messages}

    workflow.add_node("alpha", run_alpha)
    workflow.add_node("epsilon", run_epsilon)
    
    logger.debug(f"build_workflow modality='{modality}'")
    
    if modality == "spatial":
        logger.debug("Adding Beta node for spatial modality!")
        workflow.add_node("beta", run_beta)
        workflow.add_edge(START, "alpha")
        workflow.add_edge("alpha", "epsilon")
        workflow.add_edge("epsilon", "beta")
        workflow.add_edge("beta", END)
    else:
        # factorized routing
        logger.debug(f"Bypassing Beta for modality={modality}")
        workflow.add_edge(START, "alpha")
        workflow.add_edge("alpha", "epsilon")
        workflow.add_edge("epsilon", END)
    
    return workflow.compile()
