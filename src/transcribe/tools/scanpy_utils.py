import scanpy as sc
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transcribe.config import logger

def extract_top_degs(adata, cluster_col: str, cluster_id: str, top_n: int = 50) -> List[str]:
    """Retrieve top differentially expressed genes for a specific cluster."""
    # Ensure cluster_col is categorical for scanpy
    if not isinstance(adata.obs[cluster_col].dtype, pd.CategoricalDtype):
        adata.obs[cluster_col] = adata.obs[cluster_col].astype(str).astype('category')

    compute_degs = True
    if 'rank_genes_groups' in adata.uns:
        params = adata.uns['rank_genes_groups'].get('params', {})
        if params.get('groupby') == cluster_col:
            compute_degs = False
            
    if compute_degs:
        try:
            # Precompute DEG if not already done.
            sc.tl.rank_genes_groups(adata, cluster_col, method='t-test')
        except Exception as e:
            logger.error(f"Failed to compute DEGs via sc.tl.rank_genes_groups: {e}")
            return []
    
    if 'rank_genes_groups' not in adata.uns or 'names' not in adata.uns['rank_genes_groups']:
        logger.error(f"DEGs not found in adata.uns['rank_genes_groups'] for {cluster_col}")
        return []

    names = adata.uns['rank_genes_groups']['names']
    
    cid_str = str(cluster_id)
    if cid_str not in names.dtype.names:
        logger.warning(f"Cluster ID {cid_str} not found in the DEGs for column {cluster_col}.")
        return []

    return names[cid_str][:top_n].tolist()

def get_expression_profile(adata, cluster_col: str, cluster_id: str, genes: List[str]) -> Dict[str, float]:
    """Retrieve the mean expression of specific genes for a given cluster."""
    cluster_mask = adata.obs[cluster_col] == cluster_id
    
    if not any(cluster_mask):
         raise ValueError(f"No cells found for cluster {cluster_id} in {cluster_col}.")
         
    # Ensure genes are in adata
    valid_genes = [g for g in genes if g in adata.var_names]
    cluster_adata = adata[cluster_mask, valid_genes]
    
    mean_expr = np.array(cluster_adata.X.mean(axis=0)).flatten()
    return {gene: float(expr) for gene, expr in zip(valid_genes, mean_expr)}

def build_nichecard(adata, cluster_col: str, target_cluster: str, spatial_col: str = "spatial", k: int = 20) -> Dict[str, float]:
    """
    Build a spatial nichecard representing the frequency of neighboring clusters for a single cell cluster.
    """
    # 1. Ensure spatial graphs are computed
    if "spatial_connectivities" not in adata.obsp:
        try:
             # Squidpy handles spatial neighbors natively, but scanpy.pp.neighbors 
             # on spatial coordinates also works. Let's try sq.gr.spatial_neighbors
             import squidpy as sq
             if "spatial" in adata.obsm:
                 sq.gr.spatial_neighbors(adata)
             else:
                 return {"error": "No spatial coordinates found."}
        except ImportError:
             try:
                 sc.pp.neighbors(adata, use_rep=spatial_col, key_added="spatial", n_neighbors=k)
             except Exception:
                 return {}
        except Exception:
             return {}
             
    # 2. Extract the connectivities matrix
    connectivities = adata.obsp["spatial_connectivities"]
    
    # 3. Find indices for cells belonging to the target cluster
    cluster_mask = adata.obs[cluster_col] == target_cluster
    target_indices = np.where(cluster_mask)[0]
    
    if len(target_indices) == 0:
        return {}
        
    # 4. Find all neighbors of these cells
    neighbor_indices = []
    # Using scipy sparse matrix properties
    for idx in target_indices:
        # Get the row for this cell
        row = connectivities.getrow(idx)
        # Add non-zero indices (the neighbors)
        neighbor_indices.extend(row.indices)
        
    if not neighbor_indices:
        return {}
        
    # 5. Get the cluster identities of these neighbors
    neighbor_clusters = adata.obs[cluster_col].iloc[neighbor_indices]
    
    # 6. Exclude cells that belong to the SAME target cluster
    neighbor_clusters = neighbor_clusters[neighbor_clusters != target_cluster]
    
    if len(neighbor_clusters) == 0:
        return {"None": 1.0}
        
    # 7. Compute frequencies
    counts = neighbor_clusters.value_counts()
    frequencies = counts / counts.sum()
    
    return {str(k): round(float(v), 4) for k, v in frequencies.items()}
