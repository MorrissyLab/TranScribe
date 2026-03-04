
import sys
import os
import time

sys.path.append(os.path.abspath("src"))

from transcribe.workflow.graph import build_workflow
from transcribe.config import setup_logging

if __name__ == "__main__":
    setup_logging()
    print("DEBUG: STDOUT IS WORKING", flush=True)
    
    app = build_workflow(provider="gemini", model_name="gemma-3-27b-it", modality="factorized")
    
    state_input = {
        "cluster_id": "factor_1",
        "metadata": {"organism": "Human", "tissue_type": "Sarcoma", "disease": "Cancer"},
        "top_degs": ["TP53", "CD3E", "CD4"],
        "expression_profile": {"TP53": 1.0, "CD3E": 2.0, "CD4": 3.0}
    }
    
    print("DEBUG: Calling app.invoke", flush=True)
    t0 = time.time()
    try:
        res = app.invoke(state_input)
        print(f"DEBUG: app.invoke returned after {time.time()-t0:.2f}s", flush=True)
        print(res.get("final_annotation"))
    except Exception as e:
        print(f"DEBUG: Exception: {e}", flush=True)

