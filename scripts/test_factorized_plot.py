import sys
import json
import scanpy as sc
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append("src")
from transcribe.evaluation.plotting import plot_evaluation_results
from transcribe.evaluation.report_generator import generate_html_report
from transcribe.tools.factor_utils import load_factorized_data

def main():
    out_dir = Path("results/infer_results/Sarcoma_cNMF_k10_inference_gemma-3-27b-it")
    if not out_dir.exists():
        print("Run dir not found.")
        return
        
    with open(out_dir / "eval_report.json", "r") as f:
        eval_data = json.load(f)
        
    factorized_df = load_factorized_data("data/factorized/moh_sarcoma_cnmf.gene_spectra_score.k_10.dt_0_1.txt")
    usage_df = pd.read_csv("data/factorized/moh_sarcoma_cnmf.usages.k_10.dt_0_1.consensus.txt", sep="\t", index_col=0)
    adata = sc.read_h5ad("data/factorized/Sarcoma_data_Scimilarity_SAR.14223983.SYNS.Pri.s.T1.h5ad")
    
    # generate UMAP if not exist
    if "X_umap" not in adata.obsm:
        print("Computing UMAP...")
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        
    clusters = factorized_df.index.tolist()
    cluster_degs = eval_data.get("cluster_degs", {})
    predictions = {k: v.get("pred", "Unknown") for k, v in eval_data.get("cluster_mapping", {}).items()}
    
    print("Running plot_evaluation_results...")
    plot_evaluation_results(
        modality="factorized",
        adata=adata,
        clusters=clusters,
        cluster_degs=cluster_degs,
        predictions=predictions,
        factorized_type="sc",
        actual_run_name="Sarcoma_cNMF_k10_inference_gemma-3-27b-it",
        dataset_out_dir=out_dir,
        cluster_col="factor",
        eval_data=eval_data,
        is_eval=False,
        y_true=[],
        y_pred=[],
        acc=0.0,
        usage_df=usage_df
    )
    
    print("Generating HTML report...")
    generate_html_report("results/infer_results")
    print("Done")

if __name__ == "__main__":
    main()
