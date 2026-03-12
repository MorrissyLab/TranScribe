import yaml
from pathlib import Path
from transcribe.config import logger, DEFAULT_MODEL_NAME
import scanpy as sc
import pandas as pd
from transcribe.processing.inference_engine import run_analysis
from transcribe.processing.datasets import fetch_toy_dataset
from transcribe.tools.factor_utils import load_factorized_data
import re

def expand_batch_datasets(datasets: list) -> list:
    """Expands batch dataset definitions into individual datasets per rank or per file."""
    expanded = []
    for ds in datasets:
        modality = ds.get("modality", "single-cell")
        directory_path = ds.get("directory")
        
        if directory_path and modality == "factorized" and "sample_name" in ds:
            # Existing factorized batch logic
            directory = Path(directory_path)
            sample_name = ds["sample_name"]
            
            if not directory.exists():
                logger.warning(f"Batch directory not found: {directory}")
                continue
                
            rank_pattern = re.compile(rf"{re.escape(sample_name)}\.gene_spectra_score\.k_(\d+)\.")
            
            found_ranks = []
            for file_path in directory.iterdir():
                if file_path.is_file():
                    match = rank_pattern.match(file_path.name)
                    if match:
                        found_ranks.append(int(match.group(1)))
            
            if not found_ranks:
                logger.warning(f"No rank files found matching {sample_name}.gene_spectra_score.k_*.txt in {directory}")
                continue
                
            found_ranks.sort()
            logger.info(f"Found {len(found_ranks)} ranks for {sample_name}: {found_ranks}")
            
            for rank in found_ranks:
                new_ds = ds.copy()
                new_ds["name"] = f"{ds.get('name', sample_name)}_k{rank}"
                path_str = None
                for fn in directory.iterdir():
                    if fn.is_file():
                        m = rank_pattern.match(fn.name)
                        if m and int(m.group(1)) == rank:
                            path_str = str(fn)
                            break
                if not path_str: continue
                new_ds["path"] = path_str
                
                usage_pattern = re.compile(rf"{re.escape(sample_name)}\.usages\.k_{rank}\.")
                usage_str = None
                for fn in directory.iterdir():
                    if fn.is_file():
                        if usage_pattern.match(fn.name):
                            usage_str = str(fn)
                            break
                if usage_str:
                    new_ds["usage"] = usage_str
                    
                new_ds.pop("directory", None)
                new_ds.pop("sample_name", None)
                expanded.append(new_ds)

        elif directory_path and modality == "single-cell":
            # New single-cell batch logic
            directory = Path(directory_path)
            if not directory.exists():
                logger.warning(f"Batch directory not found: {directory}")
                continue
            
            found_files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix == ".h5ad"])
            
            if not found_files:
                logger.warning(f"No .h5ad files found in {directory}")
                continue
                
            logger.info(f"Found {len(found_files)} .h5ad files in {directory}")
            
            for file_path in found_files:
                new_ds = ds.copy()
                new_ds["name"] = f"{ds.get('name', 'SC')}_{file_path.stem}"
                new_ds["path"] = str(file_path)
                new_ds.pop("directory", None)
                expanded.append(new_ds)
        else:
            expanded.append(ds)
            
    return expanded

def run_yaml_eval(config_path: str, report_only: bool = False):
    import sys
    import json
    import os
    from transcribe.processing.plotting import plot_evaluation_results
    
    logger.debug(f"run_yaml_eval starting with {config_path}")
    """Parses a YAML configuration and runs run_analysis on cross products of models x datasets.
    
    The YAML should contain a 'mode' key set to 'eval', 'infer', or 'report'.
    Defaults to 'eval' if not specified.
    """
    p = Path(config_path)
    if not p.exists():
        logger.error(f"YAML config not found at {config_path}")
        return
        
    try:
        with open(p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse YAML {config_path}: {e}")
        return

    mode = config.get("mode", "eval")
    # Both CLI flag and YAML mode can trigger report-only
    is_report = (mode == "report") or report_only
    is_infer = mode == "infer"
        
    # Provider can be specified globally or inferred per model
    global_provider = config.get("provider")
    models = config.get("models", [])
    if not models:
        logger.info(f"No models specified in YAML. Falling back to default model: {DEFAULT_MODEL_NAME}")
        models = [DEFAULT_MODEL_NAME]
        
    datasets = config.get("datasets", [])
    if not datasets:
        logger.warning("No datasets specified in YAML.")
        return
        
    datasets = expand_batch_datasets(datasets)
    if not datasets:
        logger.warning("No datasets to process after evaluating batch configurations.")
        return
        
    default_output = "results/infer_results" if is_infer else "results/eval_results"
    output_base = config.get("output", default_output)
    num_tries = config.get("num_tries", 1)
    
    if is_report:
        logger.info(f"Starting report-only regeneration for {len(models)} models and {len(datasets)} datasets.")
        for model in models:
            for ds in datasets:
                ds_name = ds.get("name", "UnnamedDataset")
                run_name = f"{ds_name}_{model.replace('/', '-')}"
                dataset_out_dir = Path(output_base) / run_name
                report_path = dataset_out_dir / "eval_report.json"
                
                if not report_path.exists():
                    logger.warning(f"Skipping {run_name}: No eval_report.json found in {dataset_out_dir}")
                    continue
                
                logger.info(f"Regenerating plots for {run_name}...")
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                    
                    # Load data for plotting
                    data_path = ds.get("path", "")
                    modality = ds.get("modality", "single-cell")
                    
                    adata = None
                    usage_df = None
                    factorized_df = None
                    
                    if modality == "factorized":
                        if data_path:
                            factorized_df = load_factorized_data(data_path)
                        usage_path = ds.get("usage", None)
                        if usage_path:
                            try:
                                usage_df = pd.read_csv(usage_path, sep="\t", index_col=0)
                                # Detect orientation: if factors are columns, transpose so factors are rows
                                if usage_df.shape[1] < usage_df.shape[0]:
                                    usage_df = usage_df.T
                            except: pass
                        
                        raw_data_path = ds.get("raw_data_path")
                        if raw_data_path and os.path.exists(raw_data_path):
                             adata = sc.read_h5ad(raw_data_path)
                    else:
                        if data_path and os.path.exists(data_path):
                            adata = sc.read_h5ad(data_path)
                            # Ensure UMAP coords if missing (e.g. from Seurat)
                            from transcribe.tools.scanpy_utils import ensure_umap_coords
                            ensure_umap_coords(adata)

                    plot_evaluation_results(
                        modality=modality,
                        adata=adata,
                        clusters=eval_data.get("clusters", list(eval_data.get("cluster_mapping", {}).keys())),
                        cluster_degs=eval_data.get("cluster_degs", {}),
                        predictions=eval_data.get("inference_results", {}),
                        factorized_type=ds.get("factorized_type", "sc"),
                        actual_run_name=run_name,
                        dataset_out_dir=dataset_out_dir,
                        cluster_col=ds.get("cluster_col", "leiden" if modality != "factorized" else "factor"),
                        eval_data=eval_data,
                        is_eval=not is_infer,
                        y_true=eval_data.get("metrics", {}).get("y_true", []), # Might be missing, but plot can handle it
                        y_pred=eval_data.get("metrics", {}).get("y_pred", []),
                        acc=eval_data.get("metrics", {}).get("accuracy", 0.0),
                        usage_df=usage_df
                    )
                except Exception as e:
                    logger.error(f"Failed to regenerate report for {run_name}: {e}")
    else:
        mode_label = "inference" if is_infer else "evaluation"
        logger.info(f"Starting YAML multi-model {mode_label} for {len(models)} models and {len(datasets)} datasets. Tries per cluster: {num_tries}")
        
        for model in models:
            for ds in datasets:
                ds_name = ds.get("name", "UnnamedDataset")
                # We append the model name so the UI sees them side by side
                run_name = f"{ds_name}_{model.replace('/', '-')}"
                
                logger.info(f"{'Annotating' if is_infer else 'Evaluating'} -> Model: {model} | Dataset: {ds_name}")
                
                try:
                    data_path = ds.get("path", "")
                    modality = ds.get("modality", "single-cell")
                    
                    if data_path.lower() == "toy_data":
                        if modality == "spatial":
                            from transcribe.processing.datasets import fetch_spatial_toy_dataset
                            adata, c_col, t_col = fetch_spatial_toy_dataset()
                        else:
                            adata, c_col, t_col = fetch_toy_dataset()
                        cluster_col = ds.get("cluster_col", c_col)
                        truth_col = None if is_infer else ds.get("ground_truth_col", t_col)
                        factorized_df = None
                        usage_df = None
                        raw_data_path = None
                        factorized_type = "sc"
                    else:
                        if modality == "factorized":
                            factorized_df = load_factorized_data(data_path)
                            adata = None
                            cluster_col = ds.get("cluster_col", "factor")
                            truth_col = None if is_infer else ds.get("ground_truth_path", None)
                            raw_data_path = ds.get("raw_data_path", None)
                            
                            usage_path = ds.get("usage", None)
                            if usage_path:
                                try:
                                    usage_df = pd.read_csv(usage_path, sep="\t", index_col=0)
                                    # Detect orientation: if factors are columns, transpose
                                    if usage_df.shape[1] < usage_df.shape[0]:
                                        usage_df = usage_df.T
                                except Exception as e:
                                    logger.warning(f"Failed to load usage data from {usage_path}: {e}")
                                    usage_df = None
                            else:
                                usage_df = None
                                
                            factorized_type = ds.get("factorized_type", "sc")
                        else:
                            adata = sc.read_h5ad(data_path)
                            factorized_df = None
                            usage_df = None
                            cluster_col = ds.get("cluster_col", "leiden")
                            truth_col = None if is_infer else ds.get("ground_truth_col", None)
                            raw_data_path = None
                            factorized_type = "sc"
                        
                    from transcribe.core.llm_factory import LLMFactory
                    model_provider = global_provider or LLMFactory.infer_provider(model)

                    run_analysis(
                        adata=adata,
                        factorized_df=factorized_df,
                        usage_df=usage_df,
                        raw_data_path=raw_data_path,
                        data_path=data_path,
                        cluster_col=cluster_col,
                        ground_truth_col=truth_col,
                        dataset_name=ds_name,
                        run_name=run_name,
                        provider=model_provider,
                        model_name=model,
                        out_dir=output_base,
                        organism=ds.get("organism", "Human"),
                        tissue=ds.get("tissue", "Unknown"),
                        disease=ds.get("disease", "Normal"),
                        num_tries=num_tries,
                        modality=modality,
                        factorized_type=factorized_type
                    )
                except Exception as e:
                    logger.error(f"Error during {mode_label} of {run_name}: {e}")
                
    # After all models and datasets are run, generate a single HTML report 
    # to compare them head to head
    try:
        from transcribe.processing.report_generator import generate_html_report
        generate_html_report(output_base)
    except Exception as e:
        logger.error(f"Error generating overall HTML report: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TranScribe multiple-model evaluation pipeline")
    parser.add_argument("config", nargs="?", default="test_factorized.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    logger.info(f"Running evaluation with config: {args.config}")
    run_yaml_eval(args.config)
