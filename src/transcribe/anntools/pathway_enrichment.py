from types import SimpleNamespace
from typing import Collection, Literal, Optional, Union
import os
import pandas as pd
import numpy as np
from gprofiler import GProfiler

from transcribe.config import logger
from .utils import clean_mixed_gene_names
from .pl import save_pathway_enrichment_plots

def run_topics_pathway_enrichment(
    rf_usages: pd.DataFrame, 
    gene_set: str, 
    results_dir_path: str, 
    top_n_features: int = 1000, 
    genome: str = 'mm10', 
    experiment_title: str = "experiment"
):
    """
    Run pathway enrichment analysis for topics in `rf_usages` dataframe.
    
    Saves results as CSV tables and PDF plots.
    """
    gene_set_safe = gene_set.replace(":", "_")
    rf_usages.index = clean_mixed_gene_names(rf_usages.index.tolist(), genome)

    try:
        result = program_gprofiler(
            program_df=rf_usages,
            species="hsapiens" if genome == 'GRCh38' else 'mmusculus',
            n_hsg=top_n_features,
            gene_sets=[gene_set],
            min_termsize=10,
            max_termsize=2000
        )
    except Exception as e:
        logger.error(f"Error in program_gprofiler: {e}")
        if len(rf_usages.index) < 500:
            logger.warning("Expression matrix might be too small for effective enrichment.")
        return

    output_path = os.path.join(results_dir_path, "pathway_enrichment_results")
    os.makedirs(output_path, exist_ok=True)
    file_base = f"{experiment_title}_{gene_set_safe}_n{top_n_features}"

    # Save CSV enrichment results
    result.gprofiler_output.to_csv(os.path.join(output_path, f"results_{file_base}.csv"))
    result.summary.to_csv(os.path.join(output_path, f"results_summary_{file_base}.csv"))

    # Prepare readable annotation summary
    annotation_summary = {}
    for topic in rf_usages.columns:
        if ("-log10pval", topic) in result.summary.columns:
            sorted_pathways = result.summary[("-log10pval", topic)].dropna().sort_values(ascending=False)
            annotation_summary[topic] = [f"{x[2]}:{score:.2f}:{x[1]}" for x, score in zip(sorted_pathways.index, sorted_pathways.values)]
    
    annotation_df = pd.DataFrame({k: pd.Series(v) for k, v in annotation_summary.items()})
    annotation_df.to_csv(os.path.join(output_path, f"readable_summary_{file_base}.csv"))
    logger.info(f"Saved pathway enrichment tables to {output_path}")

    save_pathway_enrichment_plots(result, results_dir_path, gene_set, top_n_features, experiment_title)

def program_gprofiler(
    program_df: pd.DataFrame,
    species: Literal["hsapiens", "mmusculus"],
    n_hsg: int = 1000,
    gene_sets: Collection[str] = [],
    min_termsize: int = 10,
    max_termsize: int = 2000,
    batch_size: int = 20,
    show_progress_bar: bool = True
) -> SimpleNamespace:
    """Run g:Profiler enrichment for multiple programs."""
    result = SimpleNamespace()
    result.background = program_df.dropna(how="all").index.tolist()
    
    prog_names_str = program_df.columns.map(str)
    if program_df.columns.nlevels == 1:
        prog_names_str_decoder = {progstr: [prog] for progstr, prog in zip(prog_names_str, program_df.columns)}
    else:
        prog_names_str_decoder = {progstr: list(prog) for progstr, prog in zip(prog_names_str, program_df.columns)}
    prog_level_names = program_df.columns.names

    result.query = {}
    for program in program_df.columns:
        result.query[program] = program_df[program].sort_values(ascending=False).head(n_hsg).index.tolist()

    gp = GProfiler(return_dataframe=True)
    g_results = []
    
    queries = list(result.query.keys())
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_query = {str(q): result.query[q] for q in batch}
        
        batch_result = gp.profile(
            organism=species,
            query=batch_query,
            sources=gene_sets,
            no_iea=False,
            domain_scope="annotated",
            measure_underrepresentation=False,
            no_evidences=False,
            user_threshold=0.05,
            significance_threshold_method="g_SCS"
        )
        g_results.append(batch_result)
        
    result.gprofiler_output = pd.concat(g_results) if g_results else pd.DataFrame()
    if result.gprofiler_output.empty:
        raise ValueError("No enrichment results found.")

    result.gprofiler_output["-log10pval"] = np.minimum(-np.log10(result.gprofiler_output["p_value"]), 10)
    
    subset = (result.gprofiler_output["term_size"] <= max_termsize) & (result.gprofiler_output["term_size"] >= min_termsize)
    result.summary = result.gprofiler_output[subset].pivot(index=["source", "native", "name", "description", "term_size"], columns="query")
    
    stats = ["-log10pval", "query_size", "intersection_size"]
    result.summary = result.summary[stats]
    result.summary.columns = pd.MultiIndex.from_tuples(
        [([c[0]] + prog_names_str_decoder[c[1]]) for c in result.summary.columns], 
        names=["stat"] + prog_level_names
    )

    # Reorder columns to match input
    if program_df.columns.nlevels == 1:
        sorted_cols = pd.MultiIndex.from_tuples([(stat, prog) for stat in stats for prog in program_df.columns])
    else:
        sorted_cols = pd.MultiIndex.from_tuples([tuple([stat] + list(prog)) for stat in stats for prog in program_df.columns])
    
    result.summary = result.summary.reindex(columns=sorted_cols)
    return result
