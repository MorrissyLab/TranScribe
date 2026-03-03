import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_df_heatmap(df: pd.DataFrame, title: str, x_label: str, y_label: str, output_dir: str, cmap: str = "Blues", is_cluster: bool = False):
    """General function to plot and save a heatmap or clustermap."""
    fig_size = [10 + df.shape[1]/4, 10 + df.shape[0]/6]
    os.makedirs(output_dir, exist_ok=True)
    
    plot_data = df.astype(float).fillna(0)
    
    if is_cluster and df.shape[0] > 1 and df.shape[1] > 1:
        g = sns.clustermap(plot_data, cmap=cmap, col_cluster=True, row_cluster=True, figsize=fig_size, xticklabels=True, yticklabels=True)
        g.ax_heatmap.set_title(title)
        g.ax_heatmap.set_xlabel(x_label)
        g.ax_heatmap.set_ylabel(y_label)
        g.ax_heatmap.tick_params(axis='y', labelrotation=0, labelright=True, labelleft=False)
        for _, spine in g.ax_heatmap.spines.items():
            spine.set_visible(True)
            spine.set_color('#aaaaaa')
        g.savefig(os.path.join(output_dir, f"clustermap_{title}.pdf"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
        sns.heatmap(plot_data, cmap=cmap, ax=ax, xticklabels=True, yticklabels=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('#aaaaaa')
        fig.savefig(os.path.join(output_dir, f"heatmap_{title}.pdf"))
        plt.close()

def plot_geneset_pval_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    axlegend: Optional[plt.Axes] = None,
    cmap: str = "Blues",
    vmin: float = 0.,
    vmax: float = 10.,
    plot_title: str = "heatmap",
    show_geneset_names: bool = False
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """Plot gene set p-value heatmap."""
    if ax is None:
        width = 10 + df.shape[1]/4 if show_geneset_names else 0.5 + df.shape[1]/4
        height = 0.3 * df.shape[0] if show_geneset_names else 8
        fig, ax_plot = plt.subplots(figsize=[width, height], layout="constrained")
    else:
        ax_plot = ax
        fig = None
        
    if axlegend is None:
        figlegend, axlegend_plot = plt.subplots(figsize=[1, 3], layout="constrained")
    else:
        axlegend_plot = axlegend
        figlegend = None
    
    if not df.empty:    
        sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax_plot, cbar_ax=axlegend_plot, yticklabels=show_geneset_names)
        ax_plot.tick_params(labelleft=False, labelright=show_geneset_names, labeltop=False, labelbottom=True)
        ax_plot.set_xlabel("")
        ax_plot.set_ylabel("")
        ax_plot.set_title(plot_title)
        for spine in ax_plot.spines.values():
            spine.set_visible(True)
            spine.set_color('#aaaaaa')

    return fig, figlegend

def plot_geneset_pval_clustermap(
    df: pd.DataFrame,
    cmap: str = "Blues",
    plot_title: str = "heatmap",
    is_cluster: bool = True,
    vmin: float = 0.,
    vmax: float = 10.
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure]]:
    """Plot more detailed heatmap and clustermap for gene set p-values."""
    if df.empty:
        return None, None, None

    df_reset = df.reset_index()
    # Assuming standard gprofiler output index structure
    if "source" in df_reset.columns and "name" in df_reset.columns:
        pathways_list = df_reset["source"] + "-" + df_reset["name"]
    else:
        pathways_list = df.index.map(str)
        
    figsize = [10 + df.shape[1]/4, 10 + df.shape[0]/4]

    fig, ax_plot = plt.subplots(figsize=figsize, layout="constrained")
    figlegend, ax_leg = plt.subplots(figsize=[1, 3], layout="constrained")
    
    sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax_plot, cbar_ax=ax_leg, yticklabels=pathways_list)
    ax_plot.set(title=plot_title, xlabel="Program", ylabel="")
    ax_plot.tick_params(axis='y', labelrotation=0, labelright=True, labelleft=False)
    ax_plot.set_yticklabels([label.get_text()[:55] for label in ax_plot.get_yticklabels()])

    fig_cluster = None
    if is_cluster:
        fig_cluster_obj = sns.clustermap(df, cmap=cmap, col_cluster=True, row_cluster=False, vmin=vmin, vmax=vmax, 
                                        figsize=figsize, yticklabels=pathways_list)
        fig_cluster = fig_cluster_obj.fig
        fig_cluster_obj.ax_heatmap.set(title=plot_title, xlabel="Program", ylabel="")
        fig_cluster_obj.ax_heatmap.tick_params(axis='y', labelright=True, labelleft=False)
        fig_cluster_obj.ax_heatmap.set_yticklabels([label.get_text()[:55] for label in fig_cluster_obj.ax_heatmap.get_yticklabels()])
        fig_cluster_obj.cax.set_visible(False)

    return fig, fig_cluster, figlegend

def order_genesets(df: pd.DataFrame) -> pd.DataFrame:
    """Order genesets by the column with highest significance."""
    if df.empty:
        return df
    stats = pd.DataFrame({"col": df.idxmax(axis=1), "max": df.max(axis=1)})
    ordered = []
    for col in df.columns:
        ordered.append(stats[stats["col"] == col].sort_values("max", ascending=False))
    return df.loc[pd.concat(ordered).index]

def save_pathway_enrichment_plots(result, results_dir_path: str, gene_set: str, top_n_features: int, experiment_title: str):
    """Orchestrate saving of enrichment plots."""
    from transcribe.config import logger
    output_path = os.path.join(results_dir_path, "pathway_enrichment_results")
    os.makedirs(output_path, exist_ok=True)
    gene_set_safe = gene_set.replace(":", "_")
    file_base = f"{experiment_title}_{gene_set_safe}_n{top_n_features}"

    df_plot = result.summary["-log10pval"].dropna(how="all").fillna(0)
    df_plot = order_genesets(df_plot)
    
    try:
        # P-value Heatmap
        fig, figlegend = plot_geneset_pval_heatmap(df=df_plot, plot_title=file_base)
        if fig and figlegend:
            figlegend.savefig(os.path.join(output_path, f"heatmap_legend_{file_base}.pdf"))
            fig.savefig(os.path.join(output_path, f"heatmap_plot_{file_base}.pdf"))
            plt.close(fig)
            plt.close(figlegend)

        # Clustermap
        fig_detailed, fig_cluster, figlegend_c = plot_geneset_pval_clustermap(df=df_plot, plot_title=file_base)
        if fig_detailed:
            fig_detailed.savefig(os.path.join(output_path, f"heatmap_detailed_{file_base}.pdf"))
            plt.close(fig_detailed)
        if fig_cluster:
            fig_cluster.savefig(os.path.join(output_path, f"clustermap_{file_base}.pdf"))
            plt.close(fig_cluster)
        
        plt.close("all")
        logger.info(f"Saved pathway enrichment plots to {output_path}")

    except Exception as e:
        logger.warning(f"Could not render pathway enrichment plots: {e}")
        plt.close("all")
