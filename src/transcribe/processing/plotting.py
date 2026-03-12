import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import json
import os
from transcribe.config import logger
from typing import List, Dict, Any, Optional

try:
    import distinctipy
    HAS_DISTINCTIPY = True
except ImportError:
    HAS_DISTINCTIPY = False

def plot_evaluation_results(
    modality,
    adata,
    clusters,
    cluster_degs,
    predictions,
    factorized_type,
    actual_run_name,
    dataset_out_dir,
    cluster_col,
    eval_data,
    is_eval,
    y_true,
    y_pred,
    acc,
    usage_df=None
):
    """
    Generates appropriate visualization plots for the evaluation run based on the modality.
    Also handles mapping unique generated colors to the evaluation JSON payload for consistent HTML report styling.
    """
    cluster_colors = {}
    
    # Pre-generate distinct colors if possible
    if HAS_DISTINCTIPY:
        if modality == "factorized":
            # For factorized mode, we colors factors
            n_colors = len(clusters)
            if n_colors > 0:
                colors = distinctipy.get_colors(n_colors, rng=42)
                hex_colors = [distinctipy.get_hex(c) for c in colors]
                cluster_colors = {str(f): hex_colors[i] for i, f in enumerate(clusters)}
        else:
            # For categorical clusters
            if cluster_col in adata.obs:
                adata.obs[cluster_col] = adata.obs[cluster_col].astype('category')
                categories = adata.obs[cluster_col].cat.categories
                n_colors = len(categories)
                colors = distinctipy.get_colors(n_colors, rng=42)
                hex_colors = [distinctipy.get_hex(c) for c in colors]
                adata.uns[f'{cluster_col}_colors'] = hex_colors
                cluster_colors = {str(cat): hex_colors[i] for i, cat in enumerate(categories)}
    
    # 1. Prediction Mapping / Factor Loadings
    if modality == "factorized":
        if adata is not None:
            try:
                plots_dir = dataset_out_dir / "plots"
                os.makedirs(plots_dir, exist_ok=True)
                
                for cid_str in clusters:
                    logger.info(f"Generating plot for factor {cid_str}...")
                    score_col = f"Factor_{cid_str}"
                    
                    # Robust index matching (str vs int)
                    idx_match = None
                    if usage_df is not None:
                        if cid_str in usage_df.index:
                            idx_match = cid_str
                        elif str(cid_str).isdigit() and int(cid_str) in usage_df.index:
                            idx_match = int(cid_str)
                    
                    if idx_match is not None:
                        # Use exact usage weights
                        weights = usage_df.loc[idx_match]
                        # Match usage cells/spots index to adata.obs index
                        common_cells = weights.index.intersection(adata.obs_names)
                        adata.obs[score_col] = 0.0
                        
                        if len(common_cells) > 0:
                            adata.obs.loc[common_cells, score_col] = weights.loc[common_cells].values
                        else:
                            logger.warning(f"No matching cells/spots between usage and adata for factor {cid_str}")
                    else:
                        # Fallback to mean of top DEGs
                        top_genes = cluster_degs.get(cid_str, [])
                        valid_genes = [g for g in top_genes if g in adata.var_names]
                        if valid_genes:
                            if hasattr(adata.X, "toarray"):
                                 adata.obs[score_col] = adata[:, valid_genes].X.mean(axis=1).A1
                            else:
                                 adata.obs[score_col] = adata[:, valid_genes].X.mean(axis=1)
                        else:
                            logger.warning(f"No valid genes to compute score for factor {cid_str}")
                            continue

                    if factorized_type == "spatial":
                        import squidpy as sq
                        try:
                            lib_id = list(adata.uns['spatial'].keys())[0]
                        except Exception:
                            lib_id = None
                        fig, ax = plt.subplots()
                        sc.pl.spatial(adata, color=score_col, ax=ax, library_id=lib_id, show=False)
                        plot_filename = f"factor_{cid_str}_usage.png"
                    else:
                        # Ensure UMAP coordinates exist
                        if 'X_umap' not in adata.obsm:
                            logger.warning(f"No UMAP coordinates found for {cid_str}. Attempting to compute UMAP...")
                            try:
                                if 'neighbors' not in adata.uns:
                                    sc.pp.neighbors(adata)
                                sc.tl.umap(adata)
                            except Exception as e:
                                logger.error(f"Failed to compute UMAP for {cid_str}: {e}")
                                continue
                        
                        fig, ax = plt.subplots()
                        sc.pl.umap(adata, color=score_col, ax=ax, show=False)
                        plot_filename = f"factor_{cid_str}_usage.png"
                    
                    plt.title(f"Factor {cid_str}")
                    plt.tight_layout()
                    plt.savefig(f"{plots_dir}/{plot_filename}", bbox_inches="tight")
                    plt.close()
            except Exception as e:
                logger.warning(f"Failed to generate factor visualizations: {e}")
        else:
            logger.info("Skipping visualizations for factorized mode because raw_data_path/adata was not provided.")
    else:
        try:
            logger.debug(f"Generating {'Spatial Plot' if modality == 'spatial' else 'UMAP'}...")
            
            # Colors are already handled in pre-generation block above if HAS_DISTINCTIPY is True
            # Check if we should use spatial plotting
            has_spatial = 'spatial' in adata.uns or modality == "spatial"
            
            if has_spatial:
                logger.debug("Generating Spatial Plot...")
                import squidpy as sq
                try:
                    lib_id = list(adata.uns['spatial'].keys())[0] if 'spatial' in adata.uns else None
                except Exception:
                    lib_id = None
                    
                fig, ax = plt.subplots()
                # Use sc.pl.spatial as a robust alternative or sq if available
                try:
                    sc.pl.spatial(adata, color=cluster_col, ax=ax, library_id=lib_id, show=False)
                    plot_filename = "spatial_predicted.png"
                except Exception as e:
                    logger.warning(f"sc.pl.spatial failed, trying sq.pl.spatial_scatter: {e}")
                    sq.pl.spatial_scatter(adata, color=cluster_col, shape=None, ax=ax, library_id=lib_id, show=False)
                    plot_filename = "spatial_predicted.png"
            else:
                logger.debug("Generating UMAP...")
                sc.pl.umap(adata, color=cluster_col, show=False, legend_loc='on data')
                plot_filename = "umap_predicted.png"
            
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
            
    # 2. Match Count Graph for Evaluation Mode
    if is_eval:
        logger.debug("Generating match count plot...")
        df = pd.DataFrame({"Cluster": [str(c) for c in clusters], "True_Label": y_true, "Predicted": y_pred})
        df['Exact_Match'] = df['True_Label'] == df['Predicted']
        
        plt.figure()
        sns.countplot(data=df, x='Exact_Match')
        plt.title(f"Annotation Exact Matches (Acc: {acc:.2f})")
        plt.savefig(f"{dataset_out_dir}/match_plot.png")
        plt.close()
    else:
        df = pd.DataFrame({"Cluster": [str(c) for c in clusters], "Predicted": [predictions.get(str(c), "Unknown") for c in clusters]})
        
    return df
