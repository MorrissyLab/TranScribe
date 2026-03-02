import os
import json
import scanpy as sc
import matplotlib.pyplot as plt

def generate_mock_data(dataset_name="PBMC3k_EvalMock", is_eval=True):
    dataset_dir = f"eval_results/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Try umap
    try:
        adata = sc.datasets.pbmc3k_processed()
        # Mock predicted label
        adata.obs['predicted_label'] = adata.obs['louvain']
        sc.pl.umap(adata, color='predicted_label', show=False, legend_loc=None)
        plt.title(dataset_name)
        plt.savefig(f"{dataset_dir}/umap_predicted.png", bbox_inches="tight")
        plt.close()
    except Exception:
        pass
        
    cluster_mapping = {}
    if is_eval:
        cluster_mapping = {
            "0": {"true": "CD4 T cells", "pred": "CD4 T cells"},
            "1": {"true": "B cells", "pred": "Unknown"},
            "2": {"true": "Megakaryocytes", "pred": "Platelets"}
        }
    else:
        cluster_mapping = {
            "0": {"pred": "CD4 T cells"},
            "1": {"pred": "Unknown"},
            "2": {"pred": "Platelets"}
        }
        
    cluster_degs = {
        "0": ["IL7R", "CD3E", "CD3D", "LCK", "CD4"],
        "1": ["MS4A1", "CD79A", "CD79B", "BANK1", "IGHD"],
        "2": ["PPBP", "PF4", "GNG11", "NRGN", "TUBB1"]
    }

    raw_results = {
        "0": {"confidence": 0.95, "reasoning_chain": "T-cell markers observed."},
        "1": "Error",
        "2": {"confidence": 0.88, "reasoning_chain": "Platelet markers observed."}
    }
    
    eval_res = {}
    if is_eval:
        eval_res = {
            "0": {"is_match": True, "explanation": "Perfect match."},
            "1": {"is_match": False, "explanation": "Prediction failed."},
            "2": {"is_match": True, "explanation": "Platelets and Megakaryocytes are closely related."}
        }

    eval_data = {
        "dataset_name": dataset_name,
        "metadata": {
            "start_time": "2026-03-02T10:00:00.000",
            "end_time": "2026-03-02T10:05:30.000",
            "duration_seconds": 330.5,
            "model_name": "gemma-3-4b-it",
            "data_path": f"data/{dataset_name.lower()}.h5ad",
            "is_toy": True if "toy" in dataset_name.lower() else False,
            "is_eval": is_eval,
            "organism": "Human",
            "tissue": "PBMC",
            "disease": "Normal"
        },
        "metrics": {"accuracy": 0.33 if is_eval else 0.0, "evaluator_accuracy": 0.66 if is_eval else 0.0},
        "cluster_mapping": cluster_mapping,
        "cluster_degs": cluster_degs,
        "raw_results": raw_results,
        "evaluator_results": eval_res,
        "cluster_colors": {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}
    }

    with open(f"{dataset_dir}/eval_report.json", "w") as f:
        json.dump(eval_data, f, indent=4)
        
    print(f"Generated mock data for {dataset_name} (Eval Mode: {is_eval})")

if __name__ == "__main__":
    generate_mock_data("PBMC_Evaluation", is_eval=True)
    generate_mock_data("PBMC_Inference", is_eval=False)
    
    from transcribe.evaluation.report_generator import generate_html_report
    print("Generating HTML report...")
    generate_html_report("eval_results")
    print("Done.")
