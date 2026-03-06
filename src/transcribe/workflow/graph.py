from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from transcribe.core.schema import AgentState
from transcribe.agents.alpha_molecular import create_alpha_agent
from transcribe.agents.beta_spatial import create_beta_agent
from transcribe.agents.gamma_ontologist import create_gamma_agent
from transcribe.tools.rag.retriever import retrieve_rag_context
from transcribe.config import logger, DEFAULT_MODEL_NAME

def build_workflow(provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, modality: str = "single-cell", use_rag: bool = False, rag_index: str = ""):
    """
    Constructs the LangGraph workflow.
    - single-cell: Alpha -> Gamma
    - spatial: Alpha -> Beta -> Gamma
    """
    alpha = create_alpha_agent(provider=provider, model_name=model_name)
    beta = create_beta_agent(provider=provider, model_name=model_name)
    gamma = create_gamma_agent(provider=provider, model_name=model_name)

    workflow = StateGraph(AgentState)

    def run_alpha(state: AgentState):
        meta = state.get("metadata", {})
        
        # Build dynamic data payload to omit empty fields
        data_parts = []
        if state.get("expression_profile"):
            data_parts.append(f"Expression Profile: {state['expression_profile']}")
            
        mo = state.get("marker_overlap")
        if mo and str(mo).lower() != "none available":
            data_parts.append(f"Marker Overlap (Top 3 Genesets): {mo}")
            
        pe = state.get("pathway_enrichment")
        if pe and str(pe).lower() != "none available":
            data_parts.append(f"Pathway Enrichment (Top 10 GO terms): {pe}")
            
        data_payload = "\n".join(data_parts)

        result = alpha.invoke({
            "organism": meta.get("organism", "Unknown"),
            "tissue_type": meta.get("tissue_type", "Unknown"),
            "disease": meta.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "top_degs": state["top_degs"],
            "data_payload": data_payload
        })
        messages = state.get("messages", [])
        messages.append({"agent": "Alpha", "input": {"cluster_id": state["cluster_id"], "top_degs": state["top_degs"]}, "output": result.dict() if hasattr(result, 'dict') else str(result)})
        return {"alpha_candidates": result, "messages": messages}

    def run_beta(state: AgentState):
        meta = state.get("metadata", {})
        result = beta.invoke({
            "organism": meta.get("organism", "Unknown"),
            "tissue_type": meta.get("tissue_type", "Unknown"),
            "disease": meta.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "alpha_candidates": state.get("alpha_candidates"),
            "spatial_neighbors": state.get("spatial_neighbor_frequencies")
        })
        messages = state.get("messages", [])
        messages.append({"agent": "Beta", "input": {"alpha_candidates": str(state.get("alpha_candidates"))}, "output": str(result)})
        return {"beta_feedback": result, "messages": messages}

    def run_gamma(state: AgentState):
        logger.debug(f"Running Gamma on Cluster {state['cluster_id']}...")
        beta_feedback = state.get("beta_feedback", "None")
        metadata = state.get("metadata", {})
        
        rag_context = "None"
        if use_rag and rag_index:
            try:
                # Query RAG using top DEGs and metadata filters
                query = f"Cluster characterized by top DEGs: {', '.join(state['top_degs'][:20])}"
                filters = {}
                if metadata.get("organism") and metadata.get("organism").lower() != "unknown":
                    filters["organism"] = metadata.get("organism")
                if metadata.get("tissue") and metadata.get("tissue").lower() != "unknown":
                    filters["tissue"] = metadata.get("tissue")
                    
                rag_context = retrieve_rag_context(
                    query=query,
                    metadata_filters=filters,
                    index_name=rag_index,
                    embeddings_provider=provider
                )
            except Exception as e:
                logger.error(f"RAG Retrieval failed during Gamma execution: {e}")
                rag_context = f"RAG failed: {e}"
        
        result = gamma.invoke({
            "organism": metadata.get("organism", "Unknown"),
            "tissue_type": metadata.get("tissue", "Unknown"),
            "disease": metadata.get("disease", "Unknown"),
            "cluster_id": state["cluster_id"],
            "top_degs": state["top_degs"],
            "alpha_candidates": state["alpha_candidates"].model_dump_json(),
            "beta_feedback": beta_feedback,
            "rag_context": rag_context
        })
        
        # Log communication trace
        input_log = {
            "agent": "Gamma",
            "type": "input",
            "content": f"Alpha: {state['alpha_candidates'].model_dump_json()} | Beta: {beta_feedback} | RAG: {rag_context}"
        }
        output_log = {
            "agent": "Gamma",
            "type": "output",
            "content": result.model_dump()
        }
        state["messages"].extend([input_log, output_log])
        
        return {"final_annotation": result, "messages": state["messages"]}

    workflow.add_node("alpha", run_alpha)
    workflow.add_node("gamma", run_gamma)
    
    logger.debug(f"build_workflow modality='{modality}'")
    
    if modality == "spatial":
        logger.debug("Adding Beta node for spatial modality!")
        workflow.add_node("beta", run_beta)
        workflow.add_edge(START, "alpha")
        workflow.add_edge("alpha", "beta")
        workflow.add_edge("beta", "gamma")
    else:
        # single-cell routing
        logger.debug("Single-cell routing, bypassing Beta.")
        workflow.add_edge(START, "alpha")
        workflow.add_edge("alpha", "gamma")

    workflow.add_edge("gamma", END)
    
    return workflow.compile()
