import yaml
from pathlib import Path
from transcribe.config import logger
import scanpy as sc
from transcribe.evaluation.evaluator import evaluate_dataset, fetch_toy_dataset

def run_yaml_eval(config_path: str):
    """Parses a YAML configuration and runs evaluate_dataset on cross products of models x datasets."""
    p = Path(config_path)
    if not p.exists():
        logger.error(f"Evaluation YAML config not found at {config_path}")
        return
        
    try:
        with open(p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse YAML {config_path}: {e}")
        return
        
    provider = config.get("provider", "gemini")
    models = config.get("models", [])
    if not models:
        logger.warning("No models specified in YAML.")
        return
        
    datasets = config.get("datasets", [])
    if not datasets:
        logger.warning("No datasets specified in YAML.")
        return
        
    output_base = config.get("output", "eval_results")
    num_tries = config.get("num_tries", 1)
    
    logger.info(f"Starting YAML multi-model evaluation for {len(models)} models and {len(datasets)} datasets. Tries per cluster: {num_tries}")
    
    for model in models:
        for ds in datasets:
            ds_name = ds.get("name", "UnnamedDataset")
            # We append the model name so the UI sees them side by side
            run_name = f"{ds_name}_{model.replace('/', '-')}"
            
            logger.info(f"Evaluating -> Model: {model} | Dataset: {ds_name}")
            
            try:
                data_path = ds.get("path", "")
                if data_path.lower() == "toy_data":
                    adata, c_col, t_col = fetch_toy_dataset()
                    cluster_col = ds.get("cluster_col", c_col)
                    truth_col = ds.get("ground_truth_col", t_col)
                else:
                    adata = sc.read_h5ad(data_path)
                    cluster_col = ds.get("cluster_col", "leiden")
                    truth_col = ds.get("ground_truth_col", None)
                    
                evaluate_dataset(
                    adata=adata,
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
                    num_tries=num_tries
                )
            except Exception as e:
                logger.error(f"Error during eval of {run_name}: {e}")
                
    # After all models and datasets are run, generate a single HTML report 
    # to compare them head to head
    try:
        from transcribe.evaluation.report_generator import generate_html_report
        generate_html_report(output_base)
    except Exception as e:
        logger.error(f"Error generating overall HTML report: {e}")
