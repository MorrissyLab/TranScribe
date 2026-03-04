import click
import scanpy as sc
from pathlib import Path
from transcribe.config import logger, DEFAULT_MODEL_NAME, setup_logging
from transcribe.evaluation.yaml_runner import run_yaml_eval

@click.command()
@click.option('--config', default=None, type=str, help='Path to YAML config (set mode: eval/infer in the YAML).')
@click.option('--data_path', required=False, help='Path to the .h5ad dataset file.')
@click.option('--cluster_col', default='leiden', help='Column in adata.obs containing cluster IDs.')
@click.option('--output', default='results/infer_results', help='Directory to save results.')
@click.option('--dataset_name', default=None, type=str, help='Name for this dataset (used in report). Defaults to filename stem.')
@click.option('--use_rag', is_flag=True, help='Enable RAG context retrieval for Agent Gamma.')
def cli(config: str, data_path: str, cluster_col: str, output: str, dataset_name: str, use_rag: bool):
    """TranScribe: Automated Annotation of Transcriptomics via Tri-Agent Framework."""
    
    # YAML config mode (reads mode, provider, model, metadata from YAML)
    if config:
        setup_logging()
        run_yaml_eval(config)
        return
        
    if not data_path:
        raise click.UsageError("Missing argument: provide --data_path or --config.")
        
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(out_dir / "run.log"))
    
    logger.info(f"Starting TranScribe inference for {data_path}")
    logger.info(f"Model: {DEFAULT_MODEL_NAME}")
    
    # Load data
    try:
        adata = sc.read_h5ad(data_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    if cluster_col not in adata.obs:
         logger.error(f"Cluster column '{cluster_col}' not found in adata.obs.")
         return

    ds_name = dataset_name or Path(data_path).stem
    run_name = f"{ds_name}_{DEFAULT_MODEL_NAME.replace('/', '-')}"
    
    from transcribe.evaluation.evaluator import evaluate_dataset
    
    evaluate_dataset(
        adata=adata,
        data_path=data_path,
        cluster_col=cluster_col,
        ground_truth_col=None,
        dataset_name=ds_name,
        run_name=run_name,
        provider="gemini",
        model_name=DEFAULT_MODEL_NAME,
        out_dir=output,
    )
    
    from transcribe.evaluation.report_generator import generate_html_report
    generate_html_report(output)
    
    logger.info(f"Inference complete. Results saved to {out_dir}")

if __name__ == "__main__":
    cli()
