
import scanpy as sc
from transcribe.config import logger

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
