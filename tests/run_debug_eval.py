
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from transcribe.evaluation.yaml_runner import run_yaml_eval
from transcribe.config import setup_logging

if __name__ == "__main__":
    setup_logging()
    print("DEBUG: STDOUT IS WORKING", flush=True)
    run_yaml_eval("configs/eval_factorized_config.yaml")
