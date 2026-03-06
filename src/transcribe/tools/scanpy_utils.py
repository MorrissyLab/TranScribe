import scanpy as sc
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transcribe.config import logger


def get_all_degs(adata, cluster_col: str, top_n: int = 50) -> Dict[str, List[str]]:
    """Compute DEGs once for all clusters and return a mapping of {cluster_id: [genes]}."""
    
    # 1. Ensure cluster column is categorical
    if not isinstance(adata.obs[cluster_col].dtype, pd.CategoricalDtype):
        adata.obs[cluster_col] = adata.obs[cluster_col].astype(str).astype('category')
    
    # 2. Compute DEGs if not already present for this column
    compute_degs = True
    if 'rank_genes_groups' in adata.uns:
        params = adata.uns['rank_genes_groups'].get('params', {})
        if params.get('groupby') == cluster_col:
            compute_degs = False
    
    if compute_degs:
        logger.debug(f"Computing DEGs for {cluster_col}...")
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sc.tl.rank_genes_groups(adata, cluster_col, method='t-test', use_raw=True)
        except Exception as e:
            logger.error(f"Failed to compute DEGs: {e}")
            return {}

    # 3. Extract all groups
    results = {}
    groups = adata.uns['rank_genes_groups']['names'].dtype.names
    
    # Check if indices are numeric (indices vs symbols)
    # We check the first cluster's first few genes
    sample_genes = adata.uns['rank_genes_groups']['names'][0]
    is_numeric = False
    if len(sample_genes) > 0:
        # Check first 5 genes of the first cluster
        first_group = groups[0]
        check_genes = [str(adata.uns['rank_genes_groups']['names'][i][first_group]) for i in range(min(5, len(adata.uns['rank_genes_groups']['names'])))]
        if all(g.isdigit() for g in check_genes):
            is_numeric = True
            logger.debug(f"DEGs for {cluster_col} appear to be numeric indices. Mapping to gene symbols...")

    for group in groups:
        # get_rank_genes_groups_df is convenient but might be slow in a loop if called many times
        # but here we are calling it per group (cluster), which is fine.
        df = sc.get.rank_genes_groups_df(adata, group=group)
        if df is None:
            results[str(group)] = []
            continue
            
        genes = df['names'].head(top_n).tolist()
        logger.debug(f"Group {group}: Extracted {len(genes)} raw gene entries (top_n={top_n})")
        
        if is_numeric:
            indices = []
            for g in genes:
                try:
                    indices.append(int(g))
                except (ValueError, TypeError):
                    continue
            
            # Map to var names
            # If rank_genes_groups used raw, we should map to raw.var_names
            if adata.uns['rank_genes_groups']['params'].get('use_raw', True) and adata.raw is not None:
                mapped_genes = [str(adata.raw.var_names[i]) for i in indices if i < len(adata.raw.var_names)]
            else:
                mapped_genes = [str(adata.var.index[i]) for i in indices if i < len(adata.var)]
            results[str(group)] = mapped_genes
        else:
            results[str(group)] = [str(g) for g in genes]
            
    return results

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

def ensure_umap_coords(adata):
    """Ensures X_umap is present in adata.obsm, falling back to other names if necessary."""
    if "X_umap" in adata.obsm:
        return
        
    if "X_umap.rna" in adata.obsm:
        logger.info("Found X_umap.rna. Using it as X_umap.")
        adata.obsm["X_umap"] = adata.obsm["X_umap.rna"]
    elif "X_umap.atac" in adata.obsm:
        logger.info("Found X_umap.atac. Using it as X_umap.")
        adata.obsm["X_umap"] = adata.obsm["X_umap.atac"]
    else:
        logger.info("X_umap missing or coordinates not found. Computing UMAP...")
        try:
            if "X_pca" not in adata.obsm:
                sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        except Exception as e:
            logger.warning(f"Failed to compute UMAP: {e}")
