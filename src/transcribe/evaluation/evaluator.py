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
from transcribe.tools.scanpy_utils import extract_top_degs, get_expression_profile
from sklearn.metrics import accuracy_score
from transcribe.config import logger, DEFAULT_MODEL_NAME

# Import factored out modules
from transcribe.evaluation.datasets import fetch_toy_dataset, fetch_spatial_toy_dataset
from transcribe.evaluation.plotting import plot_evaluation_results

from transcribe.anntools.marker_overlap import compute_genesets_annotation
from transcribe.anntools.pathway_enrichment import run_topics_pathway_enrichment

def evaluate_dataset(adata=None, factorized_df=None, raw_data_path: str = None, cluster_col: str = "leiden", ground_truth_col: str = None, dataset_name: str = "PBMC3kToy", run_name: str = None, provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, out_dir: str = "results/eval_results", organism: str = "Human", tissue: str = "PBMC", disease: str = "Normal", data_path: str = "toy_data", num_tries: int = 1, modality: str = "single-cell", factorized_type: str = "sc"):
    """
    Evaluates the TranScribe framework against a dataset (or runs inference if ground_truth_col is None).
    """
    print(f"DEBUG: Entering evaluate_dataset for {dataset_name} (modality={modality})", flush=True)
    start_time_iso = datetime.now().isoformat()
    start_ts = time.time()
    
    actual_run_name = run_name if run_name else dataset_name
    dataset_out_dir = Path(out_dir) / actual_run_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    app = build_workflow(provider=provider, model_name=model_name, modality=modality)
    delta_agent = create_delta_agent(provider=provider, model_name=model_name)
    
    from transcribe.tools.factor_utils import extract_top_factor_markers
    
    # Check if we are in Evaluation Mode or Inference Mode
    is_eval = False
    clusters = []
    
    if modality == "factorized":
        if factorized_df is None:
            raise ValueError("factorized_df must be provided for modality='factorized'")
        clusters = factorized_df.index.tolist()
        if ground_truth_col is not None:
             is_eval = True
             
        if raw_data_path and os.path.exists(raw_data_path):
             logger.info(f"Loading raw data for factorized visualization from {raw_data_path}")
             adata = sc.read_h5ad(raw_data_path)
    else:
        if adata is None:
             raise ValueError("adata must be provided for non-factorized modality")
        if cluster_col not in adata.obs:
             raise ValueError(f"Cluster column '{cluster_col}' missing in adata.obs")
        if ground_truth_col and ground_truth_col in adata.obs:
             is_eval = True
        if is_eval and ground_truth_col not in adata.obs:
             raise ValueError(f"Ground truth column '{ground_truth_col}' missing in adata.obs")
        clusters = adata.obs[cluster_col].unique()

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
    
    from transcribe.core.llm_factory import LLMFactory
    from transcribe.tools.scanpy_utils import build_nichecard
    import json as json_builtin
    
    # Run anntools for factorized mode
    marker_overlap_df = None
    pathway_enrichment_df = None
    print(f"DEBUG: Checking if modality is factorized and factorized_df is provided (modality={modality})", flush=True)
    if modality == "factorized" and factorized_df is not None:
        try:
            print("DEBUG: Inside factorized block, beginning anntools...", flush=True)
            # We use a default geneset for marker overlap and pathway enrichment
            # You can make these configurable later
            geneset_name = "mouse_celltypes_all" if "mouse" in organism.lower() else "human_cancer_all"
            
            # 1. Run Marker Overlap
            print(f"DEBUG: Running compute_genesets_annotation for {geneset_name}", flush=True)
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
                print(f"DEBUG: Found overlap csv: {overlap_csv}", flush=True)
                marker_overlap_df = pd.read_csv(overlap_csv, index_col=0)
                
            # 2. Run Pathway Enrichment
            genome = "mm10" if "mouse" in organism.lower() else "GRCh38"
            print(f"DEBUG: Running run_topics_pathway_enrichment for GO:BP with genome {genome}", flush=True)
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
                print(f"DEBUG: Found pathway csv: {pathway_csv}", flush=True)
                pathway_enrichment_df = pd.read_csv(pathway_csv, index_col=0)
                
        except Exception as e:
            logger.error(f"Error running anntools for factorized mode: {e}")
            print(f"DEBUG: Caught exception in anntools: {e}", flush=True)
            
    print(f"DEBUG: Finished anntools block. Clusters: {len(clusters)}", flush=True)
    
    for cluster_id in clusters:
        # Rate limit protection
        time.sleep(2)
        cid_str = str(cluster_id)
        logger.info(f"{'Evaluating' if is_eval else 'Annotating'} cluster {cid_str}" + (f" (Truth: {true_labels[cid_str]})" if is_eval else ""))
        
        try:
            nichecard = {}
            if modality == "factorized":
                top_degs, expr_profile = extract_top_factor_markers(factorized_df, cid_str, top_n=50)
                cluster_degs[cid_str] = top_degs
            else:
                top_degs = extract_top_degs(adata, cluster_col, cid_str)
                cluster_degs[cid_str] = top_degs
                expr_profile = get_expression_profile(adata, cluster_col, cid_str, top_degs)
                
                if modality == "spatial":
                    nichecard = build_nichecard(adata, cluster_col, cid_str)
                    logger.info(f"Spatial Nichecard for {cid_str}: {nichecard}")
            
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
#                if pathway_enrichment_df is not None and cid_str in pathway_enrichment_df.columns:
#                    # Pathway enrichment CSV stores string representations, we will parse them back or just pass as strings
#                    # Since schema expects Dict[str, float] for pathway_enrichment, and the CSV has format "term:score:name"
#                    # let's extract it
#                    top_pathways = {}
#                    pathway_strings = pathway_enrichment_df[cid_str].dropna().head(10).tolist()
#                    for p_str in pathway_strings:
#                        try:
#                            # format is usually ID:score:Name
#                            parts = p_str.split(":", 2)
#                            if len(parts) >= 2:
#                                top_pathways[parts[0] + " " + (parts[2] if len(parts)>2 else "")] = float(parts[1])
#                        except Exception:
#                            pass
#                    if top_pathways:
#                        state_input["pathway_enrichment"] = top_pathways
            candidate_anns = []
            final_states = []
            
            for attempt in range(num_tries):
                if attempt > 0:
                    time.sleep(2)
                print(f"DEBUG: Invoking app for cluster {cid_str} (attempt {attempt+1})", flush=True)
                final_state = app.invoke(state_input)
                print(f"DEBUG: App finished for cluster {cid_str}", flush=True)
                final_states.append(final_state)
                if final_state.get("final_annotation"):
                    candidate_anns.append(final_state.get("final_annotation"))
            
            print(f"DEBUG: Found {len(candidate_anns)} candidates for cluster {cid_str}", flush=True)
            ann = None
            if len(candidate_anns) == 1:
                ann = candidate_anns[0]
                final_state = final_states[0]
            elif len(candidate_anns) > 1:
                llm = LLMFactory.get_provider(provider).get_llm(model_name, temperature=0.1)
                sys_msg = "You are an expert Ontologist. You are given several candidate cell type annotations. Select the best one."
                usr_msg = f"Tissue: {tissue}\nDisease: {disease}\n\nCandidates:\n"
                for idx, c in enumerate(candidate_anns):
                    usr_msg += f"[{idx+1}] Cell Type: {c.cell_type}\nConfidence: {c.confidence}\nReasoning: {c.reasoning_chain}\n\n"
                usr_msg += "Respond ONLY with the EXACT 'Cell Type' string of the best candidate from the list above. Do not include any other text."
                
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    if "gemma" in model_name.lower():
                        combined_msg = HumanMessage(content=f"{sys_msg}\n\n{usr_msg}")
                        resp = llm.invoke([combined_msg])
                    else:
                        resp = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
                    best_cell_type = resp.content.strip()
                    ann = next((c for c in candidate_anns if c.cell_type.lower() == best_cell_type.lower()), candidate_anns[-1])
                    final_state = next((s for s, c in zip(final_states, candidate_anns) if c == ann), final_states[-1])
                except Exception as e:
                    logger.error(f"Error resolving multiple tries: {e}")
                    ann = candidate_anns[-1]
                    final_state = final_states[-1]
            elif final_states:
                final_state = final_states[-1]
                
            traces[cid_str] = [msg.dict() if hasattr(msg, 'dict') else msg for msg in final_state.get("messages", [])] if final_state else []
            
            if ann:
                predictions[cid_str] = ann.cell_type
                raw_results[cid_str] = ann.dict() if hasattr(ann, 'dict') else ann
                if is_eval:
                    logger.info(f"Predicted: {ann.cell_type} | Truth: {true_labels[cid_str]}")
                else:
                    logger.info(f"Predicted: {ann.cell_type}")
            else:
                predictions[cid_str] = "Unknown"
        except Exception as e:
            logger.error(f"Error evaluating cluster {cid_str}: {e}")
            predictions[cid_str] = "Error"
            
    # Calculate naive metrics
    y_true = []
    y_pred = []
    if is_eval:
        y_true = [true_labels[str(c)] for c in clusters]
        y_pred = [predictions[str(c)] for c in clusters]
    
    # Agent Delta Evaluation
    eval_matches = []
    if is_eval:
        logger.info("Running Agent Delta (Evaluator) for biological reconciliation...")
        for cluster_id in clusters:
            cid_str = str(cluster_id)
            true_l = true_labels[cid_str]
            pred_l = predictions[cid_str]
            
            if pred_l in ["Error", "Unknown"]:
                delta_results[cid_str] = {"is_match": False, "explanation": f"Prediction failed: {pred_l}"}
                eval_matches.append(False)
                continue
                
            try:
                print(f"DEBUG: Invoking delta_agent for cluster {cid_str}", flush=True)
                match_res = delta_agent.invoke({"true_label": true_l, "predicted_label": pred_l})
                print(f"DEBUG: Delta finished for cluster {cid_str}: {match_res.is_match if hasattr(match_res, 'is_match') else '??'}", flush=True)
                delta_results[cid_str] = match_res.dict() if hasattr(match_res, 'dict') else match_res
                eval_matches.append(match_res.is_match)
                logger.info(f"Delta Match [{cid_str}]: {match_res.is_match} ({true_l} vs {pred_l})")
            except Exception as e:
                print(f"DEBUG: Delta error for cluster {cid_str}: {e}", flush=True)
                logger.error(f"Delta error for cluster {cid_str}: {e}")
                delta_results[cid_str] = {"is_match": False, "explanation": f"Evaluator Error: {e}"}
                eval_matches.append(False)

    acc = 0.0
    eval_acc = 0.0
    if is_eval:
        acc = accuracy_score(y_true, y_pred)
        eval_acc = sum(eval_matches) / len(eval_matches) if eval_matches else 0
        logger.info(f"Naive Accuracy Score (Exact Match): {acc:.2f}")
        logger.info(f"Evaluator Accuracy Score (Biological Match): {eval_acc:.2f}")
    
    end_ts = time.time()
    end_time_iso = datetime.now().isoformat()
    duration = end_ts - start_ts

    cluster_mapping = {}
    for c in clusters:
        cid = str(c)
        cluster_mapping[cid] = {"pred": predictions.get(cid, "Error")}
        if is_eval:
            cluster_mapping[cid]["true"] = true_labels.get(cid, "Unknown")
    
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
            "modality": modality
        },
        "metrics": {"accuracy": float(acc), "evaluator_accuracy": float(eval_acc)},
        "cluster_mapping": cluster_mapping,
        "cluster_degs": cluster_degs,
        "raw_results": raw_results,
        "evaluator_results": delta_results,
        "cluster_colors": cluster_colors
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
        acc=acc
    )
    
    # Generate interactive HTML report
    try:
        from transcribe.evaluation.report_generator import generate_html_report
        generate_html_report(out_dir)
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
    
    return acc, df_results

def run_toy_evaluation():
    logger.info("Starting Toy Evaluation Pipeline.")
    adata, cluster_col, truth_col = fetch_toy_dataset()
    evaluate_dataset(adata, cluster_col=cluster_col, ground_truth_col=truth_col, dataset_name="PBMC3kToy")
    
if __name__ == "__main__":
    run_toy_evaluation()
