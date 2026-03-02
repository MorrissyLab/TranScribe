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

def evaluate_dataset(adata, cluster_col: str, ground_truth_col: str = None, dataset_name: str = "PBMC3kToy", provider: str = "gemini", model_name: str = DEFAULT_MODEL_NAME, out_dir: str = "eval_results", organism: str = "Human", tissue: str = "PBMC", disease: str = "Normal", data_path: str = "toy_data"):
    """
    Evaluates the TranScribe framework against a dataset (or runs inference if ground_truth_col is None).
    """
    start_time_iso = datetime.now().isoformat()
    start_ts = time.time()
    
    dataset_out_dir = Path(out_dir) / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    app = build_workflow(provider=provider, model_name=model_name, modality="single-cell")
    delta_agent = create_delta_agent(provider=provider, model_name=model_name)
    
    # Check if necessary data exists
    if cluster_col not in adata.obs or ground_truth_col not in adata.obs:
         raise ValueError(f"Fields {cluster_col} or {ground_truth_col} missing in adata.obs")
         
    # To avoid Gemini Free Tier API rate limits (15 RPM), only evaluate a subset for quick testing
    clusters = adata.obs[cluster_col].unique()
    
    # Track metadata
    is_toy = "toy" in data_path.lower() or dataset_name.lower().endswith("toy")
    
    logger.info(f"Evaluating {len(clusters)} clusters for {dataset_name}...")
    
    # Map cluster ID to true label (majority vote or direct mapping)
    true_labels = {}
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
    
    for cluster_id in clusters:
        # Rate limit protection for Gemini/Gemma Free Tier (15 RPM)
        time.sleep(2)
        cid_str = str(cluster_id)
        logger.info(f"{'Evaluating' if is_eval else 'Annotating'} cluster {cid_str}" + (f" (Truth: {true_labels[cid_str]})" if is_eval else ""))
        
        try:
            top_degs = extract_top_degs(adata, cluster_col, cid_str)
            cluster_degs[cid_str] = top_degs
            expr_profile = get_expression_profile(adata, cluster_col, cid_str, top_degs)
            
            state_input = {
                 "cluster_id": cid_str,
                 "metadata": metadata,
                 "top_degs": top_degs,
                 "expression_profile": expr_profile,
                 "messages": []
            }
            final_state = app.invoke(state_input)
            ann = final_state.get("final_annotation")
            
            traces[cid_str] = final_state.get("messages", [])
            
            if ann:
                predictions[cid_str] = ann.cell_type
                raw_results[cid_str] = ann.dict() if hasattr(ann, 'dict') else ann
                logger.info(f"Predicted: {ann.cell_type} | Truth: {true_labels[cid_str]}")
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
                match_res = delta_agent.invoke({"true_label": true_l, "predicted_label": pred_l})
                delta_results[cid_str] = match_res.dict() if hasattr(match_res, 'dict') else match_res
                eval_matches.append(match_res.is_match)
                logger.info(f"Delta Match [{cid_str}]: {match_res.is_match} ({true_l} vs {pred_l})")
            except Exception as e:
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
            "disease": disease
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
        
    # Generate UMAP corresponding to Prediction
    try:
        logger.info("Generating UMAP...")
        # Map back to array, fallback to cluster ID if not annotated
        def map_pred(c):
            p = predictions.get(str(c), "")
            if p in ["", "Unknown", "Error"]:
                return f"Cluster {c}"
            return p
            
        adata.obs['predicted_label'] = adata.obs[cluster_col].apply(map_pred).astype('category')
        sc.pl.umap(adata, color='predicted_label', show=False, legend_loc=None)
        
        # Extract colors assigned by scanpy
        if 'predicted_label_colors' in adata.uns:
            categories = adata.obs['predicted_label'].cat.categories
            colors = adata.uns['predicted_label_colors']
            label_to_color = dict(zip(categories, colors))
            
            # Map back to cluster IDs for the JSON report
            for c in clusters:
                lbl = map_pred(c)
                if lbl in label_to_color:
                    cluster_colors[str(c)] = label_to_color[lbl]
                    
        # Re-save JSON with colors updated
        eval_data["cluster_colors"] = cluster_colors
        with open(dataset_out_dir / "eval_report.json", "w") as f:
            json.dump(eval_data, f, indent=4)
            
        plt.title(dataset_name)
        plt.tight_layout()
        plt.savefig(f"{dataset_out_dir}/umap_predicted.png", bbox_inches="tight")
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
