import yaml
from pathlib import Path
from transcribe.config import logger
import scanpy as sc
import pandas as pd
from transcribe.evaluation.evaluator import evaluate_dataset
from transcribe.evaluation.datasets import fetch_toy_dataset
from transcribe.tools.factor_utils import load_factorized_data

def run_yaml_eval(config_path: str):
    import sys
    print(f"DEBUG: run_yaml_eval starting with {config_path}")
    sys.stdout.flush()
    """Parses a YAML configuration and runs evaluate_dataset on cross products of models x datasets.
    
    The YAML should contain a 'mode' key set to 'eval' or 'infer'.
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
    is_infer = mode == "infer"
        
    provider = config.get("provider", "gemini")
    models = config.get("models", [])
    if not models:
        logger.warning("No models specified in YAML.")
        return
        
    datasets = config.get("datasets", [])
    if not datasets:
        logger.warning("No datasets specified in YAML.")
        return
        
    default_output = "results/infer_results" if is_infer else "results/eval_results"
    output_base = config.get("output", default_output)
    num_tries = config.get("num_tries", 1)
    
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
                        from transcribe.evaluation.datasets import fetch_spatial_toy_dataset
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
                    
                evaluate_dataset(
                    adata=adata,
                    factorized_df=factorized_df,
                    usage_df=usage_df,
                    raw_data_path=raw_data_path,
                    data_path=data_path,
                    cluster_col=cluster_col,
                    ground_truth_col=truth_col,
                    dataset_name=ds_name,
                    run_name=run_name,
                    provider=provider,
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
        from transcribe.evaluation.report_generator import generate_html_report
        generate_html_report(output_base)
    except Exception as e:
        logger.error(f"Error generating overall HTML report: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TranScribe multiple-model evaluation pipeline")
    parser.add_argument("config", nargs="?", default="test_factorized.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    print(f"Running evaluation with config: {args.config}")
    run_yaml_eval(args.config)
