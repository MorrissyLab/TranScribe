# TranScribe: An Agentic Framework for Automated Annotation of Single-cell and Spatial Transcriptomics

The goal is to automate the annotation of single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) data using Large Language Model (LLM) agents


## 1. Literature Review

The transition from traditional machine learning classifiers to multi-agent reasoning is a major focus in current computational biology literature. Your review should synthesize papers that explore the following core concepts:

* **Multi-Agent Consensus Mechanisms:** Research detailing how multiple specialized AI agents are deployed to independently evaluate the same cluster and then "debate" controversial or ambiguous cell populations until a consensus is reached.
* **Knowledge-Retrieval Integration:** Studies focused on agents that do not rely solely on internal weights, but instead actively query external biological ontologies, marker reference literature, and gene set enrichment algorithms to ground their inferences in established biology.
* **Spatial-Aware Reasoning Pipelines:** Literature on frameworks that translate physical tissue coordinates into a format an AI agent can reason about, allowing it to evaluate spatial coherence and tissue architecture alongside standard gene expression profiles.

## 2. Benchmark Dataset Curation

To rigorously evaluate your agent, find the benchmark collection that consists of datasets with high-quality, expert-curated ground truth labels.

**Single-Cell Transcriptomic Requirements:**

* **Comprehensive Reference Atlases:** Large-scale, multi-organ datasets with granular, hierarchical annotations. These are essential for testing the agent's ability to generalize across different tissue types and identify subtle cell states.
* **Complex Disease Microenvironments:** Datasets featuring high cellular heterogeneity, such as those detailing the tumor microenvironment. These are critical for benchmarking the agent's ability to accurately distinguish malignant cells from infiltrating immune or stromal cells based purely on transcriptomic signatures.

**Spatial Transcriptomic Requirements:**

* **High-Resolution Imaging Data:** Datasets with single-cell or sub-cellular resolution where ground truth is established via paired single-nucleus sequencing or expert manual annotation.
* **Sequencing-Based Spatial Grids:** Lower-resolution datasets (where spots contain multiple cells) that are the gold standard for benchmarking spatial layer clustering, anatomical region identification, and deconvolution accuracy.

## 3. Proposed Architecture: The Tri-Agent Ontology Framework

This proposed architecture simulates the rigorous peer-review process of a multidisciplinary team of computational biologists, utilizing three distinct, interacting personas.

### A. The Input & Information Hub

* **Inputs:** 
    - For scRNA-seq, Differentially expressed gene lists (ranked by log-fold change and statistical significance). 
    - For Spatial Transcriptomics, gene scores transcriptomics , maybe spatial coordinate matrices, and spatial neighborhood graphs.

* **Abstract Capabilities:** The environment provides the agents with abstract functional capabilities:
    1. Pathway enrichment analysis using gsea 
    2. marker gene sets overal from databases 
    3. mathematical similarity calculations (comparing cluster expression vectors to established reference embeddings) 
    4. Extendable to other tools and databases

### B. The Agentic Network

**1. Agent Alpha: The Molecular Analyst**

* **Role:** Analyzes the purely transcriptomic signals.
* **Action:** Ingests the top marker genes, executes gene set enrichment analysis, and retrieves established literature markers. It outputs a probabilistic list of candidate cell types or functional states based *exclusively* on the gene expression profile.
* **Output:** A list of candidate cell types or functional states with confidence scores.

**2. Agent Beta: The Spatial Contextualizer**

* **Role:** Analyzes the physical neighborhood and architectural logic.
* **Action:** Examines the spatial adjacency matrix. If Agent Alpha proposes an annotation that violates biological spatial coherence (e.g., proposing a specific epithelial cell type that is entirely isolated within a dense region of distinct neural cells), Agent Beta flags the biological implausibility and requests a re-evaluation. 
* **Resources:** According to what ? how it could decide based on what information ? 
* **Output:** A list of candidate cell types or functional states with confidence scores.

**3. Agent Gamma: The Ontologist & Critic**

* **Role:** The final decision-maker and standardizer.
* **Action:** Mediates the analysis between the Molecular Analyst or/and the Spatial Contextualizer (incase of single cell is paired with spatial data or seperately). It forces the final, agreed-upon annotation to strictly map to standardized biological nomenclature and ontologies. 
* **Output:** It outputs the final cell type or cell states, or cell activity, a calculated confidence score, and a concise, biologically grounded reasoning chain.

## 4. Evaluation Plan

To rigorously evaluate the agent, we will compare its performance against the gold-standard annotations derived from the curated benchmark datasets. The evaluation will focus on three primary axes:

### A. Accuracy Metrics

* **Cell Type Classification Accuracy:** We will calculate the precision, recall, and F1-score for each cell type, comparing the agent's predictions against the ground truth labels. This will allow us to identify specific cell types or states that the agent struggles to distinguish.
* **Spatial Coherence Score:** For spatial transcriptomics data, the metric is cosine similarity between the predicted cell type expression profile and the actual cell type expression profile. 
* **Ontological Consistency:** We will measure the degree to which the agent's annotations adhere to standardized biological ontologies. This will involve checking for the use of consistent and appropriate terminology, as well as the hierarchical correctness of the annotations.

### B. Robustness Testing

* **Sensitivity Analysis:** We will evaluate the agent's performance under varying conditions, such as different levels of input noise, varying marker gene lists, and different spatial resolutions. This will help us understand the agent's robustness to real-world experimental variability. Oncology datasets are very noisy and have batch effects, so this is important.
* **Cross-Dataset Generalization:** We will test the agent's ability to generalize to new datasets from different experimental platforms or biological contexts. This will involve evaluating the agent's performance on datasets that were not used during the training or fine-tuning process.

### C. Efficiency Analysis

* **Computational Performance:** We will measure the time and computational resources required for the agent to process different types of datasets. This will allow us to assess the practical feasibility of deploying the agent in real-world research workflows.
* **Scalability:** We will evaluate how the agent's performance and efficiency scale with increasing dataset size and complexity. This will help us understand the limitations of the current architecture and identify areas for potential optimization.

### D. Agent Evaluation

* **Human Evaluation:** We will ask domain experts to evaluate the agent's annotations and provide feedback on their biological plausibility and accuracy.
* **Automated Evaluation:** We will use automated tools to evaluate the agent's annotations and compare them against the ground truth labels. (That is implemented before in other evaluation metric here we will use it to evaluate the agent's performance)

## 5. Implementation Plan

### A. Technology Stack

* **Agent Framework:** LangChain
* **Large Language Model:** Gemini 3    
* **Transcriptomics Analysis Tools:** Scanpy
* **Data Processing:** Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn

### B. Interface 

- Option 1: CLI for the agent where specify the data path, matrices for gene expression and clustering results (for single cell) or factorization results (for spatial transcriptomics) and the agent will analyze it and output the results plots (umap for single cell with annotations and spatial transcriptomics maps with annotations).
- Option 2: UI for the agent where upload the data and the agent will analyze it and output the results and visualize umap for single cell with annotations and spatial transcriptomics maps with annotations.

