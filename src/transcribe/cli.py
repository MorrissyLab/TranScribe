import click
import json
import scanpy as sc
from pathlib import Path
from transcribe.workflow.graph import build_workflow
from transcribe.tools.scanpy_utils import extract_top_degs, get_expression_profile, build_nichecard
from transcribe.config import logger, DEFAULT_MODEL_NAME, setup_logging
from transcribe.evaluation.yaml_runner import run_yaml_eval

@click.command()
@click.option('--eval_config', default=None, type=str, help='Path to YAML config for multi-model benchmarking.')
@click.option('--data_path', required=False, help='Path to the .h5ad dataset file.')
@click.option('--modality', type=click.Choice(['single-cell', 'spatial']), default='single-cell', help='Data modality.')
@click.option('--cluster_col', default='leiden', help='Column in adata.obs containing cluster IDs.')
@click.option('--output', default='results', help='Directory to save the annotation reports.')
@click.option('--provider', default='gemini', help='LLM Provider (gemini or openai).')
@click.option('--model', default=DEFAULT_MODEL_NAME, help='GenAI model to use.')
@click.option('--organism', default='Human', help='Organism (e.g. Human, Mouse).')
@click.option('--tissue', default='Unknown', help='Tissue type (e.g. PBMC, Brain).')
@click.option('--disease', default='Normal', help='Disease state (e.g. Normal, Tumor).')
@click.option('--use_rag', is_flag=True, help='Enable RAG context retrieval for Agent Gamma.')
@click.option('--pinecone_index', default='', help='Pinecone Index to retrieve from if --use_rag is passed.')
def cli(eval_config: str, data_path: str, modality: str, cluster_col: str, output: str, provider: str, model: str, organism: str, tissue: str, disease: str, use_rag: bool, pinecone_index: str):
    """TranScribe: Automated Annotation of Transcriptomics via Tri-Agent Framework."""
    
    # Check if user wants YAML multi-model sweep
    if eval_config:
        run_yaml_eval(eval_config)
        return
        
    if not data_path:
        raise click.UsageError("Missing --data_path: You must provide either --data_path or --eval_config.")
        
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to save to the output directory
    setup_logging(log_file=str(out_dir / "run.log"))
    
    logger.info(f"Starting TranScribe for {modality} data at {data_path}")
    logger.info(f"Provider: {provider} | Model: {model}")
    logger.info(f"Metadata -> Organism: {organism}, Tissue: {tissue}, Disease: {disease}")
    if use_rag:
        logger.info(f"RAG Enabled on index: {pinecone_index}")
    
    # Load data
    logger.info(f"Loading AnnData from {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    if cluster_col not in adata.obs:
         logger.error(f"Cluster column '{cluster_col}' not found in adata.obs.")
         return
         
    clusters = adata.obs[cluster_col].unique()
    logger.info(f"Found {len(clusters)} clusters to annotate in column {cluster_col}.")
    
    # Compile workflow
    app = build_workflow(provider=provider, model_name=model, modality=modality, use_rag=use_rag, rag_index=pinecone_index)
    results = {}
    traces = {}
    
    metadata = {"organism": organism, "tissue_type": tissue, "disease": disease}
    
    for cluster_id in clusters:
        logger.info(f"Processing cluster {cluster_id}")
        try:
             # Convert to str as sometimes it can be category or int
             cid_str = str(cluster_id)
             top_degs = extract_top_degs(adata, cluster_col, cid_str)
             expr_profile = get_expression_profile(adata, cluster_col, cid_str, top_degs)
             
             state_input = {
                 "cluster_id": cid_str,
                 "metadata": metadata,
                 "top_degs": top_degs,
                 "expression_profile": expr_profile,
                 "messages": []
             }
             
             if modality == "spatial":
                 state_input["spatial_neighbor_frequencies"] = build_nichecard(adata, cluster_col, cid_str)
                 
             # Run workflow
             logger.info(f"Invoking agent network for cluster {cid_str}")
             final_state = app.invoke(state_input)
             ann = final_state.get("final_annotation")
             
             traces[cid_str] = final_state.get("messages", [])
             
             if ann:
                 results[cid_str] = ann.dict() if hasattr(ann, 'dict') else ann
                 logger.info(f"Finished {cid_str}: {ann.cell_type} (Confidence: {ann.confidence})")
             else:
                 logger.warning(f"No final annotation produced for {cid_str}")
             
        except Exception as e:
             logger.error(f"Error processing cluster {cluster_id}: {e}")
             
    # Save results
    report_path = out_dir / f"annotation_report.json"
    with open(report_path, "w") as f:
         json.dump(results, f, indent=4)
         
    trace_path = out_dir / f"communication_trace.json"
    with open(trace_path, "w") as f:
         json.dump(traces, f, indent=4)
         
    logger.info(f"Annotation complete. Report saved to {report_path}")
    logger.info(f"Communication trace saved to {trace_path}")

if __name__ == "__main__":
    cli()
