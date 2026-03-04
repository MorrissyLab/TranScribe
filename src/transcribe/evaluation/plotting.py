
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import json
from transcribe.config import logger

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
    acc
):
    """
    Generates appropriate visualization plots for the evaluation run based on the modality.
    Also handles mapping unique generated colors to the evaluation JSON payload for consistent HTML report styling.
    """
    cluster_colors = {}
    
    # 1. Prediction Mapping / Factor Loadings
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
            
    # 2. Match Count Graph for Evaluation Mode
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
        
    return df
