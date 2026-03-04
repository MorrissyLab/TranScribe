import subprocess
import sys
import yaml
import os
from pathlib import Path

def run_e2e_tests():
    """
    Automates the end-to-end execution of TranScribe evaluation pipelines using CLI and configs.
    We will create temporary configs to ensure we use only one fast LLM (gemini-2.5-flash)
    across all workflows:
    1. Eval (Single-Cell)
    2. Infer (Single-Cell)
    3. Eval (Factorized)
    """
    print("=== Starting End-to-End TranScribe Tests ===")
    
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir) # Ensure we run from project root
    
    model = "gemini-2.5-flash"
    
    configs_to_test = [
        {
            "name": "Single-Cell Evaluation",
            "config": {
                "mode": "eval",
                "provider": "gemini",
                "models": [model],
                "datasets": [{"name": "E2E_SC_Eval", "path": "toy_data", "modality": "single-cell"}],
                "output": "results/e2e_sc_eval"
            }
        },
        {
            "name": "Single-Cell Inference",
            "config": {
                "mode": "infer",
                "provider": "gemini",
                "models": [model],
                "datasets": [{"name": "E2E_SC_Infer", "path": "toy_data", "modality": "single-cell"}],
                "output": "results/e2e_sc_infer"
            }
        },
        {
            "name": "Factorized Evaluation",
            "config": {
                "mode": "eval",
                "provider": "gemini",
                "models": [model],
                "datasets": [
                    {
                        "name": "E2E_Factorized", 
                        "path": "data/GSE126049_spOT_NMF_W_28_topics_genes.tsv", 
                        "modality": "factorized",
                        "raw_data_path": "data/GSM3587002_filtered_feature_bc_matrix.h5",
                        "ground_truth_path": "data/TCell_annotation_sub.tsv",
                        "factorized_type": "spatial"
                    }
                ],
                "output": "results/e2e_factorized_eval"
            }
        }
    ]
    
    # We create a temporary config and run cli.py
    for test_case in configs_to_test:
        print(f"\n---> Running E2E Test: {test_case['name']}")
        
        # Determine if data exists for the test
        if test_case['name'] == "Factorized Evaluation":
            factor_path = test_case['config']['datasets'][0]['path']
            if not os.path.exists(factor_path):
                print(f"Skipping {test_case['name']} because test data {factor_path} is missing locally.")
                continue

        tmp_config_path = f"configs/tmp_e2e_{test_case['name'].replace(' ', '_')}.yaml"
        
        with open(tmp_config_path, "w") as f:
            yaml.dump(test_case['config'], f)
            
        try:
            # Run the CLI
            result = subprocess.run(
                [sys.executable, "-m", "transcribe.cli", "--config", tmp_config_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"FAILED: {test_case['name']}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                sys.exit(1)
            else:
                print(f"SUCCESS: {test_case['name']}")
                
        finally:
            # Clean up tmp config
            if os.path.exists(tmp_config_path):
                os.remove(tmp_config_path)

    print("\n=== End-to-End Tests Completed Successfully! ===")

if __name__ == "__main__":
    run_e2e_tests()
