# TranScribe: Architecture & Project Description

TranScribe is an LLM-powered multi-agent framework designed to automate the annotation of single-cell and spatial transcriptomics datasets. By orchestrating specialized agents driven by generative AI models like **Gemma 3** or **Gemini**, TranScribe significantly reduces the manual labor required to transition from raw structural gene clusters to deeply understood cell lineages.

## Multi-Agent Architecture

### Core Annotation Loop (Inference)
The pipeline relies on a cascading decision tree among specialized, persona-driven LLM agents:

1. **Agent Alpha (Molecular Profiler)**: Analyzes the top differentially expressed genes (DEGs) out of a marker gene table for a particular cluster, offering primary hypotheses on the structural lineage of the cluster.
2. **Agent Beta (Spatial Analyst - Optional)**: In spatial transcriptomics workflows, Beta adds geographic understanding by integrating neighbor frequencies (e.g., cell types located physically adjacent to the cluster) into a secondary "Nichecard" format.
3. **Agent Gamma (Ontologist)**: The final decision maker. Synthesizes Alpha's hypotheses and Beta's niche context to deduce the final, official cell-type annotation based on accepted biomedical ontology. It yields the definitive `FinalAnnotation` payload containing a confidence score and a chain of biological reasoning.

### Agent Delta (Evaluation & Benchmarking)
For validation purposes, TranScribe supports full evaluation benchmarks against annotated ground truth data. When triggered in this mode, a fourth agent steps in:

4. **Agent Delta (Evaluator)**: Operates independently of the prediction chain to adjudicate synonymity. Because biological cell classifications are hierarchical (e.g., "CD4+ T Cell" vs "T Cell"), a standard hard-coded string match is insufficient. Agent Delta consumes both the true label and the predicted label and outputs an intelligent boolean match (`is_match`) alongside its biological reasoning.

## Reporting & Output Modes

TranScribe dynamically outputs execution logs and rich interactive HTML graphical reports via its web generator. The tool elegantly handles two primary use-cases via an `is_eval` flag dynamically inferred from the data inputs:

### 1. Inference Mode ("Running Mode")
When run on brand new, unannotated biological datasets, the framework operates in Inference Mode. The HTML generator surfaces an uncluttered UI containing:
- Extractable, dynamic dataset UMAPs mapped with Scanpy colors.
- Interactive trace logs allowing the user to view the full JSON exchange between Agent Alpha, Beta, and Gamma for any given cluster.
- Deep-dive Annotation Reasoning directly from Agent Gamma natively displayed in both Table and Card UI layouts.

### 2. Evaluation Mode ("Benchmark Mode")
If the `--ground_truth_col` is supplied and present, the system leverages Agent Delta. The HTML UI gracefully adapts to unfold additional columns and badges:
- **Naive Accuracy**: Exact string-match benchmarking.
- **Evaluator Accuracy**: The context-aware semantic accuracy evaluated by Agent Delta.
- **Match Status**: Success indicators (✅ / ❌) mapped to specific clusters showing which lineage hypotheses succeeded or failed and why.

## Built With
- **LangGraph** & **LangChain**: Graph state orchestration and structured JSON schema tracking.
- **Scanpy**: Fast integration with conventional spatial transcriptomics libraries.
- **HTML/CSS/JS**: Client-side single-file dashboards generated automatically from Pydantic output parses.
