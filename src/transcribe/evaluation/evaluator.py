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
import matplotlib.pyplot as plt
import seaborn as sns
from transcribe.config import logger, DEFAULT_MODEL_NAME

def fetch_toy_dataset():
    """Fetches the Scanpy PBMC3k processed toy dataset for evaluation."""
    logger.info("Fetching PBMC3k processed toy dataset from Scanpy...")
    try:
         adata = sc.datasets.pbmc3k_processed()
         # Hide ground truth from the LLM by creating numeric cluster IDs
         adata.obs['blind_cluster'] = adata.obs['louvain'].cat.codes.astype(str).astype('category')
    except Exception as e:
         logger.error(f"Failed to fetch pbmc3k from scanpy datasets: {e}")
         raise
    return adata, "blind_cluster", "louvain"

def fetch_spatial_toy_dataset():
    """Fetches the Squidpy Visium H&E toy dataset for spatial evaluation."""
    logger.info("Fetching Visium H&E toy dataset from Squidpy...")
    try:
         import squidpy as sq
         adata = sq.datasets.visium_hne_adata()
         # The ground truth is 'cluster'
         adata.obs['blind_cluster'] = adata.obs['cluster'].cat.codes.astype(str).astype('category')
         # Pre-compute spatial neighbors for Agent Beta
         sq.gr.spatial_neighbors(adata)
    except ImportError:
         logger.error("Squidpy is required for the spatial toy dataset. Install it with `pip install squidpy`.")
         raise
    except Exception as e:
         logger.error(f"Failed to fetch visium dataset from squidpy: {e}")
         raise
    return adata, "blind_cluster", "cluster"

def evaluate_dataset(adata=None, factorized_df=None, raw_data_path: str = None, cluster_col: str = "leiden", ground_truth_col: str = None, dataset_name: str = "PBMC3kToy", run_name: str = None, provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, out_dir: str = "eval_results", organism: str = "Human", tissue: str = "PBMC", disease: str = "Normal", data_path: str = "toy_data", num_tries: int = 1, modality: str = "single-cell", factorized_type: str = "sc"):
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
    from transcribe.core.schema import FinalAnnotation
    from transcribe.tools.scanpy_utils import build_nichecard
    import json as json_builtin
    
    for cluster_id in clusters:
        # Rate limit protection for Gemini/Gemma Free Tier (15 RPM)
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
                 "messages": []
            }
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
                # Prompt the ontologist/LLM to select the best one
                llm = LLMFactory.get_provider(provider).get_llm(model_name, temperature=0.1)
                sys_msg = "You are an expert Ontologist. You are given several candidate cell type annotations for a cell cluster. Select the best one."
                usr_msg = f"Tissue: {tissue}\\nDisease: {disease}\\n\\nCandidates:\\n"
                for idx, c in enumerate(candidate_anns):
                    usr_msg += f"[{idx+1}] Cell Type: {c.cell_type}\\nConfidence: {c.confidence}\\nReasoning: {c.reasoning_chain}\\n\\n"
                usr_msg += "Respond ONLY with the EXACT 'Cell Type' string of the best candidate from the list above. Do not include any other text."
                
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    # For Gemma models, merge system+user into a single HumanMessage
                    if "gemma" in model_name.lower():
                        combined_msg = HumanMessage(content=f"{sys_msg}\n\n{usr_msg}")
                        resp = llm.invoke([combined_msg])
                    else:
                        resp = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
                    best_cell_type = resp.content.strip()
                    # Find the candidate matching the selected cell type
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
            
    # Calculate naive metrics (if is_eval)
    y_true = []
    y_pred = []
    if is_eval:
        y_true = [true_labels[str(c)] for c in clusters]
        y_pred = [predictions[str(c)] for c in clusters]
    
    # Agent Delta Evaluation (Biological Consistency)
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

    # Metrics calculation
    acc = 0.0
    eval_acc = 0.0
    if is_eval:
        acc = accuracy_score(y_true, y_pred)
        eval_acc = sum(eval_matches) / len(eval_matches) if eval_matches else 0
        logger.info(f"Naive Accuracy Score (Exact Match): {acc:.2f}")
        logger.info(f"Evaluator Accuracy Score (Biological Match): {eval_acc:.2f}")
    
    # Save results
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
        "metrics": {
            "accuracy": float(acc),
            "evaluator_accuracy": float(eval_acc)
        },
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
        
    # Generate Plot corresponding to Prediction
    if modality == "factorized":
        if adata is not None:
            try:
                for cid_str in clusters:
                    logger.info(f"Generating plot for factor {cid_str}...")
                    score_col = f"Factor_{cid_str}"
                    top_genes = cluster_degs.get(cid_str, [])
                    valid_genes = [g for g in top_genes if g in adata.var_names]
                    if valid_genes:
                        if hasattr(adata.X, "toarray"):
                             adata.obs[score_col] = adata[:, valid_genes].X.mean(axis=1).A1
                        else:
                             adata.obs[score_col] = adata[:, valid_genes].X.mean(axis=1)
                        
                        if factorized_type == "spatial":
                            import squidpy as sq
                            try:
                                lib_id = list(adata.uns['spatial'].keys())[0]
                            except Exception:
                                lib_id = None
                            fig, ax = plt.subplots()
                            sq.pl.spatial_scatter(adata, color=score_col, shape=None, ax=ax, library_id=lib_id, legend_loc=None)
                            plot_filename = f"spatial_factor_{cid_str}.png"
                        else:
                            sc.pl.umap(adata, color=score_col, show=False)
                            plot_filename = f"umap_factor_{cid_str}.png"
                        
                        plt.title(f"{actual_run_name} - {cid_str} ({predictions.get(cid_str,'')})")
                        plt.tight_layout()
                        plt.savefig(f"{dataset_out_dir}/{plot_filename}", bbox_inches="tight")
                        plt.close()
            except Exception as e:
                logger.warning(f"Failed to generate factor visualizations: {e}")
        else:
            logger.info("Skipping visualizations for factorized mode because raw_data_path/adata was not provided.")
    else:
        try:
            logger.info(f"Generating {'Spatial Plot' if modality == 'spatial' else 'UMAP'}...")
            import distinctipy
            # We want each cluster to have a unique distinct color
            adata.obs[cluster_col] = adata.obs[cluster_col].astype('category')
            categories = adata.obs[cluster_col].cat.categories
            colors = distinctipy.get_colors(len(categories), rng=42)
            hex_colors = [distinctipy.get_hex(c) for c in colors]
            adata.uns[f'{cluster_col}_colors'] = hex_colors
            
            if modality == "spatial":
                import squidpy as sq
                # We must specify library_id to avoid warnings if not set, or just let Squidpy find it
                try:
                    lib_id = list(adata.uns['spatial'].keys())[0]
                except Exception:
                    lib_id = None
                    
                fig, ax = plt.subplots()
                sq.pl.spatial_scatter(adata, color=cluster_col, shape=None, ax=ax, library_id=lib_id, legend_loc=None)
                plot_filename = "spatial_predicted.png"
            else:
                sc.pl.umap(adata, color=cluster_col, show=False, legend_loc=None)
                plot_filename = "umap_predicted.png"
            
            # Map back to cluster IDs for the JSON report
            label_to_color = dict(zip(categories, hex_colors))
            for c in clusters:
                if str(c) in label_to_color:
                    cluster_colors[str(c)] = label_to_color[str(c)]
                        
            # Re-save JSON with colors updated
            eval_data["cluster_colors"] = cluster_colors
            with open(dataset_out_dir / "eval_report.json", "w") as f:
                json.dump(eval_data, f, indent=4)
                
            plt.title(actual_run_name)
            plt.tight_layout()
            plt.savefig(f"{dataset_out_dir}/{plot_filename}", bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate UMAPs: {e}")
        
    # Quick Plot (if is_eval)
    if is_eval:
        logger.info("Generating match count plot...")
        df = pd.DataFrame({"Cluster": [str(c) for c in clusters], "True_Label": y_true, "Predicted": y_pred})
        df['Exact_Match'] = df['True_Label'] == df['Predicted']
        
        plt.figure()
        sns.countplot(data=df, x='Exact_Match')
        plt.title(f"Annotation Exact Matches (Acc: {acc:.2f})")
        plt.savefig(f"{dataset_out_dir}/match_plot.png")
        plt.close()
    else:
        df = pd.DataFrame({"Cluster": [str(c) for c in clusters], "Predicted": [predictions.get(str(c), "Unknown") for c in clusters]})
    
    # Generate interactive HTML report
    try:
        from transcribe.evaluation.report_generator import generate_html_report
        generate_html_report(out_dir)
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
    
    return acc, df

def run_toy_evaluation():
    logger.info("Starting Toy Evaluation Pipeline.")
    adata, cluster_col, truth_col = fetch_toy_dataset()
    evaluate_dataset(adata, cluster_col=cluster_col, ground_truth_col=truth_col, dataset_name="PBMC3kToy")
    
if __name__ == "__main__":
    run_toy_evaluation()
