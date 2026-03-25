import os
from transcribe.config import logger
import scanpy as sc
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from transcribe.workflow.graph import build_workflow
from transcribe.agents.delta_evaluator import create_delta_agent
from transcribe.tools.scanpy_utils import get_all_degs, get_expression_profile, ensure_umap_coords
from sklearn.metrics import accuracy_score
from transcribe.config import logger, DEFAULT_MODEL_NAME
from tqdm import tqdm

# Import factored out modules
from transcribe.processing.datasets import fetch_toy_dataset, fetch_spatial_toy_dataset
from transcribe.processing.plotting import plot_evaluation_results

from transcribe.anntools.marker_overlap import compute_genesets_annotation
from transcribe.anntools.pathway_enrichment import run_topics_pathway_enrichment


def run_analysis(adata=None, factorized_df=None, usage_df=None, raw_data_path: str = None, cluster_col: str = "leiden", ground_truth_col: str = None, dataset_name: str = "PBMC3kToy", run_name: str = None, provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, out_dir: str = "results/eval_results", organism: str = "Human", tissue: str = "PBMC", disease: str = "Normal", data_path: str = "toy_data", num_tries: int = 1, modality: str = "single-cell", factorized_type: str = "sc"):
    """
    Runs the TranScribe analysis (inference or evaluation) against a dataset.
    If ground_truth_col is None, it runs in inference mode.
    """
    logger.debug(f"Entering run_analysis for {dataset_name} (modality={modality})")
    start_time_iso = datetime.now().isoformat()
    start_ts = time.time()
    
    actual_run_name = run_name if run_name else dataset_name
    dataset_out_dir = Path(out_dir) / actual_run_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    app = build_workflow(provider=provider, model_name=model_name, modality=modality)
    delta_agent = create_delta_agent(provider=provider, model_name=model_name)
    
    from transcribe.tools.factor_utils import extract_top_factor_markers
    
    # Check if we are in Evaluation Mode or Inference Mode
    is_eval = ground_truth_col is not None
    clusters = []
    
    if modality == "factorized":
        if factorized_df is None:
            raise ValueError("factorized_df must be provided for modality='factorized'")
        clusters = factorized_df.index.tolist()
             
        if raw_data_path and os.path.exists(raw_data_path):
             logger.info(f"Loading raw data for factorized visualization from {raw_data_path}")
             adata = sc.read_h5ad(raw_data_path)
             if factorized_type == "sc":
                 ensure_umap_coords(adata)
    else:
        if adata is None:
            raise ValueError("adata must be provided for non-factorized modalities")
        adata.obs[cluster_col] = adata.obs[cluster_col].astype(str)
        clusters = sorted(adata.obs[cluster_col].unique())
        ensure_umap_coords(adata)

    # Track metadata
    is_toy = "toy" in data_path.lower() or dataset_name.lower().endswith("toy")
    logger.info(f"{'Evaluating' if is_eval else 'Annotating'} {len(clusters)} clusters for {dataset_name}...")
    
    # Map cluster ID to true label (majority vote or direct mapping)
    true_labels = {}
    if is_eval:
        if modality == "factorized":
            if isinstance(ground_truth_col, str) and os.path.exists(ground_truth_col):
                gt_df = pd.read_csv(ground_truth_col)
                for _, row in gt_df.iterrows():
                    true_labels[str(row.iloc[0])] = str(row.iloc[1])
        else:
            for cluster_id in clusters:
                mask = adata.obs[cluster_col] == cluster_id
                most_common = adata.obs[mask][ground_truth_col].mode().iloc[0]
                true_labels[str(cluster_id)] = str(most_common)
        
    predictions = {}
    raw_results = {}
    traces = {}
    cluster_degs = {}
    delta_results = {}
    cluster_colors = {}
    
    metadata = {"organism": organism, "tissue_type": tissue, "disease": disease}
    
    def _parse_json_block(text: str):
        cleaned = str(text).strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        if "{" in cleaned and "}" in cleaned:
            cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
        return json.loads(cleaned)
    
    from transcribe.core.llm_factory import LLMFactory
    from transcribe.tools.scanpy_utils import build_nichecard, build_umap_proximity
    import json as json_builtin
    
    # Run anntools for factorized mode
    marker_overlap_df = None
    pathway_enrichment_df = None
    logger.debug(f"Checking if modality is factorized and factorized_df is provided (modality={modality})")
    if modality == "factorized" and factorized_df is not None:
        try:
            logger.debug("Inside factorized block, beginning anntools...")
            # We use a default geneset for marker overlap and pathway enrichment
            # You can make these configurable later
            geneset_name = "mouse_celltypes_all" if "mouse" in organism.lower() else "human_cancer_all"
            
            # 1. Run Marker Overlap
            logger.debug(f"Running compute_genesets_annotation for {geneset_name}")
            try:
                compute_genesets_annotation(
                    rf_usages=factorized_df.T,
                    gene_set_name=geneset_name,
                    results_dir_path=str(dataset_out_dir),
                    experiment_title=actual_run_name,
                    ranking_method="mgs"
                )
            except Exception as e:
                logger.warning(f"Error in compute_genesets_annotation (possibly plotting): {e}")

            overlap_csv = os.path.join(dataset_out_dir, "marker_overlap_results", f"scores_{actual_run_name}_{geneset_name}.csv")
            if os.path.exists(overlap_csv):
                logger.debug(f"Found overlap csv: {overlap_csv}")
                marker_overlap_df = pd.read_csv(overlap_csv, index_col=0)
                
            # 2. Run Pathway Enrichment
            genome = "mm10" if "mouse" in organism.lower() else "GRCh38"
            logger.debug(f"Running run_topics_pathway_enrichment for GO:BP with genome {genome}")
            try:
                run_topics_pathway_enrichment(
                    rf_usages=factorized_df.T,
                    gene_set="GO:BP", # Often used as default for valid gProfiler source
                    results_dir_path=str(dataset_out_dir),
                    genome=genome,
                    experiment_title=actual_run_name
                )
            except Exception as e:
                logger.warning(f"Error in run_topics_pathway_enrichment (possibly plotting): {e}")

            pathway_csv = os.path.join(dataset_out_dir, "pathway_enrichment_results", f"readable_summary_{actual_run_name}_GO_BP_n1000.csv")
            if os.path.exists(pathway_csv):
                logger.debug(f"Found pathway csv: {pathway_csv}")
                pathway_enrichment_df = pd.read_csv(pathway_csv, index_col=0)
                
        except Exception as e:
            logger.error(f"Error running anntools for factorized mode: {e}")
            logger.debug(f"Caught exception in anntools: {e}")
            
    logger.debug(f"Finished anntools block. Clusters: {len(clusters)}")
    
    # Pre-extract all DEGs for AnnData-based modalities (non-factorized)
    all_cluster_degs = {}
    singleton_clusters = []
    singleton_clusters = []
    if modality != "factorized":
        all_cluster_degs, singleton_clusters = get_all_degs(adata, cluster_col, top_n=100)
    
    phase1_results = {}
    
    pbar = tqdm(clusters, desc=f"Phase 1 (Alpha/Epsilon/Beta) for {dataset_name}")
    for cluster_id in pbar:
        # Rate limit protection
        time.sleep(2)
        cid_str = str(cluster_id)
        status_msg = f"{'Evaluating' if is_eval else 'Annotating'} cluster {cid_str}"
        if is_eval:
            status_msg += f" (Truth: {true_labels[cid_str]})"
        
        pbar.set_description(f"{status_msg}...")
        logger.debug(status_msg)
        
        try:
            nichecard = {}
            if modality == "factorized":
                top_degs, expr_profile = extract_top_factor_markers(factorized_df, cid_str, top_n=100)
                cluster_degs[cid_str] = top_degs
            else:
                top_degs = all_cluster_degs.get(cid_str, [])
                cluster_degs[cid_str] = top_degs
                expr_profile = get_expression_profile(adata, cluster_col, cid_str, top_degs[:50])
                
                if modality == "spatial":
                    nichecard = build_nichecard(adata, cluster_col, cid_str)
                    logger.debug(f"Spatial Nichecard for {cid_str}: {nichecard}")
                elif modality == "single-cell":
                    nichecard = build_umap_proximity(adata, cluster_col, cid_str)
                    logger.debug(f"UMAP Proximity for {cid_str}: {nichecard}")
            
            state_input = {
                 "cluster_id": cid_str,
                 "metadata": metadata,
                 "top_degs": top_degs,
                 "expression_profile": expr_profile,
                 "spatial_neighbor_frequencies": nichecard,
                 "marker_overlap": None,
                 "pathway_enrichment": None,
                 "messages": []
            }
            
            if modality == "factorized":
                if marker_overlap_df is not None and cid_str in marker_overlap_df.columns:
                    top_markers = marker_overlap_df[cid_str].sort_values(ascending=False).head(3).to_dict()
                    state_input["marker_overlap"] = top_markers

            # Phase 1 Execution
            logger.info(f"[Agent Call] Phase1 Workflow (Alpha/Epsilon{'/Beta' if modality == 'spatial' else ''}) | cluster={cid_str}")
            final_state = app.invoke(state_input)
            
            # Extract Epsilon and Beta summaries to include in Phase 2
            alpha_cand = final_state.get("alpha_candidates", "None")
            if hasattr(alpha_cand, "candidates"):
                alpha_text = ", ".join([f"{c.cell_type} ({c.confidence})" for c in alpha_cand.candidates[:3]])
            else:
                alpha_text = str(alpha_cand)
                
            pathway_text = ""
            pa = final_state.get("pathway_analysis")
            if pa and hasattr(pa, "biological_summary"):
                pathway_text = f"Pathways: {', '.join(pa.top_pathways)} | States: {', '.join(pa.suggested_cell_states)}"
            else:
                pathway_text = str(pa) if pa else "None"
            
            beta_text = final_state.get("beta_feedback", "None")

            phase1_results[cid_str] = {
                "alpha": alpha_text,
                "epsilon": pathway_text,
                "epsilon_raw": pa.model_dump() if hasattr(pa, 'model_dump') else (pa.dict() if hasattr(pa, 'dict') else pa) if pa else None,
                "beta": beta_text,
                "umap_neighbors": nichecard if modality == "single-cell" else {},
                "top_degs": top_degs, # Keep full list (up to 100)
                "messages": final_state.get("messages", [])
            }
        except Exception as e:
            logger.error(f"Error in Phase 1 for cluster {cid_str}: {e}")
            phase1_results[cid_str] = {
                "alpha": "Error", "epsilon": "Error", "beta": "Error", 
                "umap_neighbors": nichecard if modality == "single-cell" else {},
                "top_degs": top_degs, "messages": []
            }

    # Batch Beta for single-cell mode: evaluate UMAP neighborhood context once across all clusters.
    if modality == "single-cell":
        logger.info("Running Batch Beta for UMAP proximity reconciliation...")
        from transcribe.agents.beta_spatial import create_beta_batch_agent
        beta_batch_agent = create_beta_batch_agent(provider=provider, model_name=model_name)
        all_clusters_context = ""
        for cid_str, res in phase1_results.items():
            all_clusters_context += (
                f"\n--- CLUSTER {cid_str} ---\n"
                f"Alpha Candidates: {res.get('alpha', 'None')}\n"
                f"UMAP Proximity: {res.get('umap_neighbors', {})}\n"
            )
        try:
            logger.info("[Agent Call] Beta (Batch UMAP) | scope=all_clusters")
            beta_batch_raw = beta_batch_agent.invoke({
                "organism": metadata.get("organism", "Unknown"),
                "tissue_type": metadata.get("tissue_type", "Unknown"),
                "disease": metadata.get("disease", "Unknown"),
                "all_clusters_context": all_clusters_context
            })
            beta_batch_map = _parse_json_block(beta_batch_raw) if isinstance(beta_batch_raw, str) else (beta_batch_raw or {})
        except Exception as e:
            logger.warning(f"Batch Beta failed for single-cell mode: {e}")
            beta_batch_map = {}
        for cid_str, res in phase1_results.items():
            cluster_beta = beta_batch_map.get(cid_str, "None")
            if isinstance(cluster_beta, dict):
                adherence = cluster_beta.get("contextual_adherence", "Unknown")
                critique = cluster_beta.get("critique", "No critique.")
                beta_text = f"Contextual Adherence: {adherence}\nCritique: {critique}"
            else:
                beta_text = str(cluster_beta)
            res["beta"] = beta_text
            res["messages"].append({
                "role": "Beta Batch UMAP Critique",
                "content": beta_text
            })

    # Phase 2: Batch Gamma
    logger.info("Running Phase 2: Batch Gamma processing...")
    from transcribe.agents.gamma_ontologist import create_gamma_agent
    gamma_agent = create_gamma_agent(provider=provider, model_name=model_name)
    
    # Construct massive ALL-CLUSTER evidence prompt
    all_clusters_evidence = ""
    for cid_str, res in phase1_results.items():
        all_clusters_evidence += f"\n--- CLUSTER {cid_str} ---\n"
        all_clusters_evidence += f"Top DEGs: {', '.join(res['top_degs'])}\n"
        all_clusters_evidence += f"Alpha (Molecular): {res['alpha']}\n"
        all_clusters_evidence += f"Epsilon (Pathways): {res['epsilon']}\n"
        all_clusters_evidence += f"Beta (Spatial/UMAP Context): {res['beta']}\n"
        
    rag_context = "None"
    
    try:
        logger.info("[Agent Call] Gamma (Batch Ontologist) | scope=all_clusters")
        batch_res = gamma_agent.invoke({
            "organism": metadata.get("organism", "Unknown"),
            "tissue_type": metadata.get("tissue", "Unknown"),
            "disease": metadata.get("disease", "Unknown"),
            "all_clusters_evidence": all_clusters_evidence,
            "rag_context": rag_context
        })
        batch_annotations = batch_res.annotations if hasattr(batch_res, "annotations") else []
    except Exception as e:
        logger.error(f"Batch Gamma entirely failed: {e}")
        batch_annotations = []
        
    # Phase 3: Per-result Confidence Assessment
    logger.info("Running Phase 3: Zeta Confidence Assessment...")
    from transcribe.agents.zeta_confidence import create_zeta_agent
    from transcribe.tools.biology_tools import query_marker_database
    
    zeta_agent = create_zeta_agent(provider=provider, model_name=model_name)
    
    for ann in batch_annotations:
        cid_str = str(ann.cluster_id)
        if cid_str not in [str(c) for c in clusters]:
            continue
            
        cluster_degs_list = phase1_results.get(cid_str, {}).get("top_degs", [])
        epsilon_genes = cluster_degs_list[:50]
        gamma_reasoning = ann.reasoning_chain if hasattr(ann, "reasoning_chain") else ""
        
        # Determine expected markers
        try:
            # Use .func to call the underlying function of the LangChain tool
            expected_markers_res = query_marker_database.func(
                ann.cell_type, 
                organism=metadata.get("organism", "Human"),
                tissue=metadata.get("tissue", "PBMC"),
                provider=provider, 
                model_name=model_name,
                gamma_reasoning=gamma_reasoning
            )
        except Exception as e:
            logger.warning(f"Marker database query failed: {e}")
            expected_markers_res = ["CD4", "CD8A", "MS4A1", "CD14"]
            
        # Invoke Zeta
        try:
             # Programmatic overlap against the same DEG evidence window used by Epsilon (top 50).
             epsilon_gene_set = {g.upper() for g in epsilon_genes}
             found_markers = [m for m in expected_markers_res if str(m).upper() in epsilon_gene_set]
             calculated_score = len(found_markers) / len(expected_markers_res) if expected_markers_res else 0.0
             
             logger.info(f"[Agent Call] Zeta | cluster={cid_str}")
             zeta_res = zeta_agent.invoke({
                 "cluster_id": cid_str,
                 "predicted_cell_type": ann.cell_type,
                 "gamma_reasoning": gamma_reasoning,
                 "expected_markers": ", ".join(expected_markers_res),
                 "observed_degs": ", ".join(epsilon_genes),
                 "programmatic_score": calculated_score 
             })
             
             # Override LLM score with programmatic one for metrics
             zeta_res.overlap_score = calculated_score
             zeta_res.observed_markers = found_markers
             
             # We store the full zeta output alongside the raw final annotation
             ann_dict = ann.model_dump() if hasattr(ann, 'model_dump') else (ann.dict() if hasattr(ann, 'dict') else ann)
             ann_dict["confidence_assessment"] = zeta_res.model_dump() if hasattr(zeta_res, 'model_dump') else (zeta_res.dict() if hasattr(zeta_res, 'dict') else zeta_res)
             
             # Update the confidence based on Zeta's programmatic score
             if calculated_score > 0.6:
                 ann.confidence = "high"
             elif calculated_score > 0.3:
                 ann.confidence = "medium"
             else:
                 ann.confidence = "low"
        except Exception as e:
             logger.warning(f"Zeta failed for cluster {cid_str}: {e}")
             ann_dict = ann.model_dump() if hasattr(ann, 'model_dump') else (ann.dict() if hasattr(ann, 'dict') else ann)
             
        predictions[cid_str] = ann.cell_type
        # Add pathway details for report rendering and detailed activity modal.
        epsilon_raw = phase1_results.get(cid_str, {}).get("epsilon_raw", None)
        epsilon_text = phase1_results.get(cid_str, {}).get("epsilon", "")
        if isinstance(epsilon_raw, dict):
            pathway_activity = dict(epsilon_raw)
            pathway_activity.setdefault("summary", epsilon_raw.get("biological_summary", ""))
            pathway_activity.setdefault("reasoning", epsilon_text or epsilon_raw.get("biological_summary", ""))
        else:
            pathway_activity = epsilon_raw
        ann_dict["pathway_activity"] = pathway_activity
        raw_results[cid_str] = ann_dict
        
        traces[cid_str] = []
        # Add the 'Cluster Evidence' summary as the very first trace item for visibility
        traces[cid_str].append({
            "role": "Cluster Input Evidence", 
            "content": f"Top 100 DEGs: {', '.join(cluster_degs_list[:20])}... (+{max(0, len(cluster_degs_list)-20)} more)"
        })

        for msg in phase1_results.get(cid_str, {}).get("messages", []):
             if isinstance(msg, dict):
                  traces[cid_str].append(msg)
             elif hasattr(msg, 'content'):
                  traces[cid_str].append({"role": getattr(msg, 'type', 'agent'), "content": msg.content})
             else:
                  traces[cid_str].append({"role": "info", "content": str(msg)})
        
        # Append specific part of Batch Gamma IO for this cluster trace
        gamma_reasoning = ann.reasoning_chain if hasattr(ann, 'reasoning_chain') else "None"
        traces[cid_str].append({
            "role": "Gamma Final Decision", 
            "content": f"Decision: {ann.cell_type}\nReasoning: {gamma_reasoning}"
        })
        
        # Append Zeta Assessment
        if "confidence_assessment" in ann_dict:
            za = ann_dict["confidence_assessment"]
            traces[cid_str].append({
                "role": "Zeta Validation",
                "content": f"Overlap Score: {za.get('overlap_score')}\nNarrative: {za.get('agreement_narrative')}"
            })
            
        traces[cid_str].append({"role": "output", "content": json_builtin.dumps(raw_results[cid_str])})
        
        pred_msg = f"Predicted {cid_str}: {ann.cell_type}"
        if is_eval:
             logger.debug(f"{pred_msg} | Truth: {true_labels.get(cid_str)}")
        else:
             logger.debug(pred_msg)
             
    # Fill in any missing predictions for clusters
    for cluster_id in clusters:
        cid_str = str(cluster_id)
        if cid_str not in predictions:
            predictions[cid_str] = "Unknown"
            raw_results[cid_str] = "Phase 2 missed this cluster."
            traces[cid_str] = traces.get(cid_str, [])
            
    # Add Global Gamma Trace
    traces["__GLOBAL_GAMMA__"] = [
        {
            "role": "Batch Gamma Input",
            "content": f"Organism: {metadata.get('organism', 'Unknown')}\n\nEvidence Summary:\n{all_clusters_evidence}"
        },
        {
            "role": "Batch Gamma Output",
            "content": batch_res.model_dump_json(indent=2) if hasattr(batch_res, 'model_dump_json') else str(batch_res)
        }
    ]

    # Calculate naive metrics
    y_true = []
    y_pred = []
    if is_eval:
        y_true = [true_labels[str(c)] for c in clusters]
        y_pred = [predictions[str(c)] for c in clusters]
    
    # Agent Delta Evaluation (Consolidated into one prompt as requested)
    eval_matches = []
    eval_input = ""
    batch_eval_res = None
    if is_eval:
        logger.info("Running Agent Delta (Evaluator) for biological reconciliation...")
        for cluster_id in clusters:
            cid_str = str(cluster_id)
            true_l = true_labels.get(cid_str, "Unknown")
            pred_l = predictions.get(cid_str, "Unknown")
            degs_ctx = ", ".join(phase1_results.get(cid_str, {}).get("top_degs", [])[:10])
            eval_input += f"Cluster {cid_str}: Predicted='{pred_l}', Ground Truth='{true_l}', Context='{degs_ctx}'\n"
            
        try:
            logger.info("[Agent Call] Delta (Batch Evaluator) | scope=all_clusters")
            batch_eval_res = delta_agent.invoke({"eval_input": eval_input})
            for ev in batch_eval_res.evaluations:
                cid = str(ev.cluster_id)
                delta_results[cid] = {
                    "is_match": ev.is_match,
                    "explanation": ev.explanation,
                    "predicted_label": ev.predicted_label,
                    "true_label": ev.true_label
                }
                eval_matches.append(ev.is_match)
                logger.debug(f"Delta Match [{cid}]: {ev.is_match} ({ev.true_label} vs {ev.predicted_label})")
        except Exception as e:
            logger.error(f"Batch Delta failed: {e}")
            for cid_str in [str(c) for c in clusters]:
                delta_results[cid_str] = {"is_match": False, "explanation": f"Evaluator Error: {e}"}
                eval_matches.append(False)
        traces["__GLOBAL_DELTA__"] = [
            {
                "role": "Batch Delta Input",
                "content": f"Evaluation Pairs:\n{eval_input}"
            },
            {
                "role": "Batch Delta Output",
                "content": batch_eval_res.model_dump_json(indent=2) if hasattr(batch_eval_res, "model_dump_json") else str(batch_eval_res)
            }
        ]

    acc = 0.0
    eval_acc = 0.0
    if is_eval:
        acc = accuracy_score(y_true, y_pred)
        eval_acc = sum(eval_matches) / len(eval_matches) if eval_matches else 0
        logger.info(f"LLMaJ Accuracy Score (Biological Match): {eval_acc:.2f}")
    
    end_ts = time.time()
    end_time_iso = datetime.now().isoformat()
    duration = end_ts - start_ts
 
    cluster_mapping = {}
    for c in clusters:
        cid = str(c)
        cluster_mapping[cid] = {"pred": predictions.get(cid, "Error")}
        if is_eval:
            cluster_mapping[cid]["true"] = true_labels.get(cid, "Unknown")
            
    # Phase 4: Post-hoc Dataset Summarization (Eta)
    logger.info("Running Phase 4: Eta Descriptor Summarization...")
    from transcribe.agents.eta_descriptor import create_eta_agent
    eta_agent = create_eta_agent(provider=provider, model_name=model_name)
    all_annotations_str = ""
    for cid_str, pred in predictions.items():
         all_annotations_str += f"Cluster {cid_str}: {pred}\n"
         
    try:
         logger.info("[Agent Call] Eta (Dataset Summarizer) | scope=all_clusters")
         eta_res = eta_agent.invoke({
             "organism": metadata.get("organism", "Unknown"),
             "tissue_type": metadata.get("tissue", "Unknown"),
             "disease": metadata.get("disease", "Unknown"),
             "all_annotations": all_annotations_str
         })
         hierarchical_summary = eta_res.model_dump() if hasattr(eta_res, 'model_dump') else (eta_res.dict() if hasattr(eta_res, 'dict') else eta_res)
         logger.debug("Eta Summarization Complete.")
    except Exception as e:
         logger.error(f"Eta Summarization failed: {e}")
         hierarchical_summary = {"groups": [], "narrative_summary": f"Failed: {e}"}
    
    eval_data = {
        "dataset_name": dataset_name,
        "metadata": {
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "duration_seconds": duration,
            "model_name": model_name,
            "data_path": data_path,
            "is_toy": is_toy,
            "is_eval": is_eval,
            "organism": organism,
            "tissue": tissue,
            "disease": disease,
            "num_tries": num_tries,
            "modality": modality,
            "factorized_type": factorized_type,
            "singleton_clusters": singleton_clusters
        },
        "metrics": {"llmaj_accuracy": float(eval_acc)},
        "cluster_mapping": cluster_mapping,
        "cluster_degs": cluster_degs,
        "raw_results": raw_results,
        "inference_results": delta_results,
        "cluster_colors": cluster_colors,
        "hierarchical_summary": hierarchical_summary
    }

    with open(dataset_out_dir / "eval_report.json", "w") as f:
        json.dump(eval_data, f, indent=4)
    with open(dataset_out_dir / "eval_communication_trace.json", "w") as f:
        json.dump(traces, f, indent=4)
        
    # Generate Plots from separated module
    df_results = plot_evaluation_results(
        modality=modality,
        adata=adata,
        clusters=clusters,
        cluster_degs=cluster_degs,
        predictions=predictions,
        factorized_type=factorized_type,
        actual_run_name=actual_run_name,
        dataset_out_dir=dataset_out_dir,
        cluster_col=cluster_col,
        eval_data=eval_data,
        is_eval=is_eval,
        y_true=y_true,
        y_pred=y_pred,
        acc=acc,
        usage_df=usage_df
    )
    
    # Generate interactive HTML report
    try:
        from transcribe.processing.report_generator import generate_html_report
        generate_html_report(out_dir)
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
    
    return acc, df_results

def run_toy_analysis():
    logger.info("Starting Toy Analysis Pipeline.")
    from transcribe.processing.datasets import fetch_toy_dataset
    adata, cluster_col, truth_col = fetch_toy_dataset()
    run_analysis(adata, cluster_col=cluster_col, ground_truth_col=truth_col, dataset_name="PBMC3kToy")
    
if __name__ == "__main__":
    run_toy_analysis()
