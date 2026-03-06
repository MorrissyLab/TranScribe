"""
exporter.py
===========
Handles exporting experiment results to different formats (e.g., CSV).
"""
import csv
from pathlib import Path
from typing import List, Dict
from transcribe.config import logger

def export_summary_to_csv(datasets: List[Dict], output_path: Path):
    """
    Exports a summary of all datasets/experiments to a single CSV file.
    
    Expected fields: 
    experiment_name, cluster, predicted cell type, ground truth, top DEGs(5), reasoning
    """
    headers = [
        "experiment_name", 
        "cluster", 
        "predicted cell type", 
        "ground truth", 
        "top DEGs(5)", 
        "reasoning"
    ]
    
    rows = []
    
    for ds in datasets:
        dataset_name = ds.get("name", "Unknown")
        is_eval = ds.get("metadata", {}).get("is_eval", False)
        
        mapping = ds.get("mapping", {})
        degs = ds.get("degs", {})
        raw_results = ds.get("raw", {})
        
        for cluster_id, m in mapping.items():
            pred_lbl = m.get("pred", "Error")
            true_lbl = m.get("true", "N/A") if is_eval else "N/A"
            
            # Get top 5 DEGs
            cluster_degs = degs.get(cluster_id, [])
            top_5_degs = ", ".join(cluster_degs[:5])
            
            # Get reasoning
            raw_ann = raw_results.get(cluster_id, {})
            if isinstance(raw_ann, str):
                reasoning = raw_ann
            elif isinstance(raw_ann, dict):
                reasoning = raw_ann.get("reasoning_chain", "N/A")
            else:
                reasoning = "N/A"
                
            rows.append({
                "experiment_name": dataset_name,
                "cluster": cluster_id,
                "predicted cell type": pred_lbl,
                "ground truth": true_lbl,
                "top DEGs(5)": top_5_degs,
                "reasoning": reasoning
            })
            
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Summary CSV exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
