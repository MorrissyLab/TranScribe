"""
exporter.py
===========
Handles exporting experiment results to different formats (e.g., CSV).
"""
import csv
import pandas as pd
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

def export_experiment_degs_to_csv(ds_data: Dict, output_path: Path):
    """
    Exports top genes/DEGs for a single experiment to a CSV file.
    Columns: Cluster, Top Genes
    """
    degs = ds_data.get("degs", {})
    modality = ds_data.get("metadata", {}).get("modality", "unknown")
    column_name = "Top Genes" if modality == "factorized" else "DEGs"
    
    rows = []
    
    for cluster_id, gene_list in degs.items():
        rows.append({
            "Cluster": cluster_id,
            column_name: ", ".join(gene_list)
        })
        
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Cluster", column_name])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Experiment genes exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export experiment CSV: {e}")

def _clean_sheet_name(name: str) -> str:
    """
    Cleans and shortens experiment names for Excel sheet tabs (max 31 chars).
    Specifically extracts the SAR sample identifier if present.
    """
    import re
    
    # Try to find the SAR.XXXXX pattern which is the core identifier
    # Example: SAR.10207009.MFH.Pri.s.T2
    # We match from SAR. until we hit a model suffix indicator (_gemma, -gemma) or common separators
    sar_match = re.search(r'(SAR\..+?)(?=[_\-]gemma|[_\-]\d|(?<!\w)[_\-]|(\.txt)?$)', name, re.IGNORECASE)
    if sar_match:
        return sar_match.group(1)[:31]
        
    # Fallback cleaning logic
    prefixes_to_remove = ["MOH_CellBender_Samples_", "MOH_CellBender_Samples", "Sarcoma_cNMF_", "Sarcoma_cNMF"]
    suffixes_to_remove = ["_gemma-3-27b-it", "-gemma-3-27b-it", "_gemma", "-gemma"]
    
    clean_name = name
    for p in prefixes_to_remove:
        if clean_name.startswith(p):
            clean_name = clean_name[len(p):]
            break
    for s in suffixes_to_remove:
        if clean_name.endswith(s):
            clean_name = clean_name[:-len(s)]
            break
            
    return clean_name.lstrip("._- ")[:31]

def export_batch_degs_to_excel(datasets: List[Dict], output_base: Path):
    """
    Exports top genes/DEGs for all experiments to a single Excel file with tabs.
    Adds the full experiment name as a header in the first row.
    """
    excel_path = output_base / "gene_results_summary.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for ds in datasets:
                name = ds.get("name", "Unknown")
                modality = ds.get("metadata", {}).get("modality", "unknown")
                identity_label = "Factor" if modality == "factorized" else "Cluster"
                
                sheet_name = _clean_sheet_name(name)
                degs = ds.get("degs", {})
                
                if not degs:
                    continue
                    
                # Create a DataFrame where columns are clusters and rows are genes
                max_len = max(len(genes) for genes in degs.values()) if degs else 0
                data = {}
                for cluster_id, genes in degs.items():
                    data[f"{identity_label} {cluster_id}"] = genes + [""] * (max_len - len(genes))
                
                df = pd.DataFrame(data)
                
                # Write to Excel starting from row 2 (0-indexed) to leave room for header
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                
                # Add Header in row 0
                worksheet = writer.sheets[sheet_name]
                worksheet.cell(row=1, column=1, value=f"Experiment: {name}")
                # Optional: format header (e.g. bold, though we'll keep it simple for now)
                
        logger.info(f"Batch genes exported to {excel_path}")
    except Exception as e:
        logger.error(f"Failed to export batch Excel: {e}")
