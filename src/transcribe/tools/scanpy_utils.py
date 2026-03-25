import pandas as pd
import scanpy as sc
import numpy as np
import warnings
import logging
from typing import List, Dict, Tuple
from transcribe.config import logger

# Note: The user manually added multiple import blocks. 
# We consolidate them here while keeping the logic intact.

def get_all_degs(adata, cluster_col: str, top_n: int = 50) -> tuple:
    """Compute DEGs once for all clusters and return a tuple of:
       - mapping of {cluster_id: [genes]}
       - list of singleton cluster IDs excluded from DEG computation
    """
    
    # 1. Ensure cluster column is categorical
    if not isinstance(adata.obs[cluster_col].dtype, pd.CategoricalDtype):
        adata.obs[cluster_col] = adata.obs[cluster_col].astype(str).astype('category')
    
    # 1b. Identify singleton clusters (< 2 cells) that crash t-test
    cluster_counts = adata.obs[cluster_col].value_counts()
    small_clusters = [str(c) for c in cluster_counts[cluster_counts < 2].index.tolist()]
    
    if small_clusters:
        logger.warning(
            f"Excluding {len(small_clusters)} singleton cluster(s) from DEG computation "
            f"(< 2 cells): {small_clusters}"
        )
        mask = ~adata.obs[cluster_col].isin(small_clusters)
        adata_for_degs = adata[mask].copy()
        adata_for_degs.obs[cluster_col] = adata_for_degs.obs[cluster_col].cat.remove_unused_categories()
    else:
        adata_for_degs = adata
    
    # 2. Compute DEGs if not already present for this column
    params = adata.uns.get('rank_genes_groups', {}).get('params', {})
    if params.get('groupby') != cluster_col:
        logger.debug(f"Computing DEGs for {cluster_col}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc.tl.rank_genes_groups(adata_for_degs, cluster_col, method='t-test', use_raw=False)
                # Copy results back to original adata
                adata.uns['rank_genes_groups'] = adata_for_degs.uns['rank_genes_groups']
                # Update params after computing
                params = adata.uns['rank_genes_groups']['params'] 
        except Exception as e:
            logger.error(f"Failed to compute DEGs: {e}")
            return {}, small_clusters

    # 3. Determine the correct gene names index based on 'use_raw'
    use_raw = params.get('use_raw', True)
    if use_raw and adata.raw is not None:
        gene_names = adata.raw.var.index
    else:
        gene_names = adata.var.index

    # 4. Extract all groups directly from the structured NumPy array
    results = {}
    names_record = adata.uns['rank_genes_groups']['names']
    
    for group in names_record.dtype.names:
        # Slice top N directly (extremely fast compared to pandas conversions)
        raw_genes = [str(g) for g in names_record[group][:top_n]]
        
        # 5. Map indices to names if they are digits AND not literally named as digits in the index
        if raw_genes and raw_genes[0].isdigit() and raw_genes[0] not in gene_names:
            results[str(group)] = [gene_names[int(idx)] for idx in raw_genes if int(idx) < len(gene_names)]
        else:
            results[str(group)] = raw_genes
    
    # 6. Add singleton clusters back with empty gene lists
    for sc_id in small_clusters:
        results[sc_id] = []
            
    return results, small_clusters

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

def build_umap_proximity(adata, cluster_col: str, target_cluster: str) -> Dict[str, float]:
    """
    Build a UMAP-based proximity map for a cluster.
    Computes centroid of each cluster in UMAP space, then returns
    proximity scores (inverse distance, normalized) from the target cluster
    to all other clusters. This serves as a single-cell analog to spatial nichecards.
    """
    ensure_umap_coords(adata)
    
    if "X_umap" not in adata.obsm:
        logger.warning("No UMAP coordinates available for proximity computation.")
        return {}
    
    umap_coords = adata.obsm["X_umap"]
    clusters = adata.obs[cluster_col].astype(str)
    unique_clusters = sorted(clusters.unique())
    
    if str(target_cluster) not in unique_clusters:
        return {}
    
    # Compute centroids for all clusters
    centroids = {}
    for cid in unique_clusters:
        mask = clusters == cid
        centroids[cid] = np.mean(umap_coords[mask], axis=0)
    
    target_centroid = centroids[str(target_cluster)]
    
    # Compute distances from target to all other clusters
    distances = {}
    for cid, centroid in centroids.items():
        if cid == str(target_cluster):
            continue
        dist = np.linalg.norm(target_centroid - centroid)
        distances[cid] = dist
    
    if not distances:
        return {}
    
    # Convert to proximity scores (inverse distance, normalized)
    max_dist = max(distances.values())
    if max_dist == 0:
        return {cid: 1.0 for cid in distances}
    
    proximity = {}
    for cid, dist in distances.items():
        proximity[cid] = round(float(1.0 - dist / max_dist), 4)
    
    # Sort by proximity (highest first)
    proximity = dict(sorted(proximity.items(), key=lambda x: x[1], reverse=True))
    
    return proximity

