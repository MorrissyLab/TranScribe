import os
import pandas as pd
import rbo
from typing import List, Optional
from transcribe.config import logger
from .utils import list_genesets, read_gmt_file
from .pl import plot_df_heatmap

def get_ranking_score(query_list: List[str], program_list: List[str], rank_type: str = 'rbo') -> float:
    """Calculate similarity score between two gene lists."""
    if not query_list or not program_list:
        return 0.0
        
    if rank_type == 'rbo':
        return rbo.RankingSimilarity(query_list, program_list).rbo()
    elif rank_type == "rboext":
        return rbo.RankingSimilarity(query_list, program_list).rbo_ext()
    elif rank_type == 'mgs':
        intersected_genes = [i + 1 for i, x in enumerate(program_list) if x in query_list]
        return sum(1 / rank for rank in intersected_genes)
    return 0.0

def compute_genesets_annotation(
    rf_usages: pd.DataFrame, 
    gene_set_name: str, 
    results_dir_path: str, 
    max_top_genes: int = 100, 
    ranking_method: str = "rboext", 
    experiment_title: str = "experiment"
):
    """
    Compute gene set overlap scores and generate outputs.
    """
    # Locate geneset file
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geneset_dir = os.path.join(parent_directory, '..', '..', 'data', 'genesets')
    if not os.path.isdir(geneset_dir):
        geneset_dir = os.path.join(parent_directory, 'data', 'genesets')

    gmt_file = os.path.join(geneset_dir, f"{gene_set_name}.gmt")

    if not os.path.isfile(gmt_file):
        available = list_genesets()
        logger.error(f"Gene set '{gene_set_name}' not found at {gmt_file}. Available: {available}")
        return

    # Read gene set
    df_geneset = read_gmt_file(gmt_file)
    output_path = os.path.join(results_dir_path, "marker_overlap_results")
    os.makedirs(output_path, exist_ok=True)
    
    # Compute scores
    logger.info(f"Computing overlap scores for {gene_set_name}...")
    df_scores = pd.DataFrame(index=df_geneset.columns, columns=rf_usages.columns, dtype=float)
    
    for q_set in df_geneset.columns: 
        query_list = df_geneset[q_set].dropna().unique().tolist()
        n_top = min(len(query_list) * 2, max_top_genes)
        
        for p_col in rf_usages.columns:
            program_list = rf_usages[p_col].sort_values(ascending=False).head(n_top).index.tolist()
            df_scores.at[q_set, p_col] = get_ranking_score(query_list, program_list, rank_type=ranking_method)

    # Save CSV
    csv_path = os.path.join(output_path, f"scores_{experiment_title}_{gene_set_name}.csv")
    df_scores.to_csv(csv_path)
    logger.info(f"Saved overlap scores to {csv_path}")

    # Plot
    try:
        plot_df_heatmap(
            df_scores, 
            title=f"{experiment_title}_{gene_set_name}", 
            x_label="Topic/Cluster", 
            y_label="Gene Set", 
            output_dir=output_path, 
            is_cluster=True
        )
        logger.info(f"Saved overlap plots to {output_path}")
    except Exception as e:
        logger.warning(f"Could not render overlap plots: {e}")
