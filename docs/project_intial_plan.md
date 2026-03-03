# TranScribe: An Agentic Framework for Automated Annotation of Single-cell and Spatial Transcriptomics

## 1. Introduction

The unprecedented advancement of highly multiplexed transcriptomic technologies has fundamentally transformed the landscape of modern biology, offering an unparalleled resolution into cellular heterogeneity, developmental trajectories, and complex tissue architectures. Single-cell RNA sequencing (scRNA-seq) has enabled the genome-wide profiling of millions of individual cells, laying the groundwork for massive, pan-organ reference atlases. Simultaneously, the rapid emergence of spatially resolved transcriptomics (ST) has introduced a critical physical dimension to this data, allowing researchers to study not only what genes are expressed, but precisely where these cells reside within intact tissue microenvironments. This spatial context is paramount for understanding cell-cell communication, functional tissue units, and the pathological restructuring of microenvironments in diseases such as cancer and neurodegeneration.

However, despite these technological leaps in data generation, the computational interpretation of this data—specifically the accurate, reproducible annotation of cell types and spatial niches—remains a profound bottleneck. Traditional analytical pipelines rely heavily on deterministic clustering algorithms followed by heuristic, manual inspection of differentially expressed genes (DEGs). This process is notoriously subjective, deeply dependent on domain-specific human expertise, and highly susceptible to batch effects and platform-specific noise. While early supervised machine learning classifiers and deep learning models sought to automate this step, they frequently suffer from an inability to generalize across distinct biological conditions or differing sequencing protocols.

The recent explosion of Large Language Models (LLMs) has introduced a promising new paradigm: the application of artificial intelligence capable of natural language reasoning, tool utilization, and autonomous logical deduction. Yet, deploying standalone, single-agent LLMs directly into bioinformatics pipelines presents significant risks, such as hallucinating marker associations or failing to grasp complex spatial geometries. To resolve these systemic challenges, the field of computational biology is rapidly pivoting toward multi-agent, collaborative reasoning frameworks. This comprehensive report outlines an exhaustive synthesis of current literature, curates a rigorous benchmarking dataset ecosystem, and proposes the architecture for **TranScribe**: a novel Tri-Agent Ontology Framework explicitly engineered to automate single-cell and spatial transcriptomic annotation.

## 2. Literature Review

The transition from isolated computational models to multi-agent reasoning networks represents the cutting edge of biological data interpretation. A synthesis of contemporary computational biology literature reveals three foundational pillars driving the evolution of transcriptomic annotation:

### Multi-Agent Consensus Mechanisms

Research details how multiple specialized AI agents are deployed to independently evaluate the same cluster and then "debate" controversial or ambiguous cell populations until a consensus is reached. Systems such as **CASSIA** and **CellAgent** have pioneered the strategic division of analytical labor among specialized, interacting LLM personas to guard against hallucinations and calibrate confidence. Similarly, the **CyteType** framework introduces a sophisticated multi-agent ecosystem utilizing hypothesis-driven reasoning, where an Annotator agent generates competing hypotheses and a Reviewer agent simulates an expert panel to perform automated reference checking. In spatial transcriptomics, frameworks like **SpatialAgent** and **NicheAgent** employ an "Analyst–Consensus–Reviewer" triad to manage the complexities of microenvironmental mapping and prevent "label collapse."

### Dynamic Knowledge-Retrieval Integration

Studies focus on agents that do not rely solely on internal weights, but instead actively query external biological ontologies, marker reference literature, and gene set enrichment algorithms to ground their inferences in established biology. Biomedical ontologies like the Gene Ontology (GO), Human Phenotype Ontology (HPO), and Cell Ontology (CL) serve as foundational scaffolds. Frameworks like **Biomni** and **BioMaster** equip LLM agents with the ability to execute API calls to external databases (e.g., NCBI, Reactome, PubMed), while **KGARevion** integrates knowledge graphs for structured semantic triplet reasoning.

### Spatial-Aware Reasoning Pipelines

Translating physical tissue coordinates into a format an AI agent can reason about allows it to evaluate spatial coherence alongside standard gene expression profiles. Two dominant approaches have emerged:

1. **Latent spatial embedding integration:** Frameworks like **SPELL** use Graph Autoencoders to compress k-nearest neighbor graphs into latent embeddings.
2. **Text-based neighborhood encoding:** Lightweight frameworks such as **NicheAgent** convert coordinates into structured semantic representations termed *nichecards*, which encode canonical marker genes, expression centroids, and neighboring cell-type frequencies within a defined radius.

## 3. Benchmark Dataset Curation

To rigorously evaluate the agent, the benchmark collection consists of datasets with high-quality, expert-curated ground truth labels spanning both multi-organ reference atlases and complex disease microenvironments.

### Single-Cell Transcriptomic Requirements

* **Comprehensive Reference Atlases:** Large-scale, multi-organ datasets with granular, hierarchical annotations (e.g., Human Cell Landscape, Tabula Sapiens, Mouse Cell Atlas) are essential for testing the agent's ability to generalize across different tissue types.
* **Complex Disease Microenvironments:** Datasets featuring high cellular heterogeneity, such as tumor microenvironments (e.g., TISCH2), are critical for benchmarking the agent's ability to accurately distinguish malignant cells from infiltrating immune or stromal cells.

| Dataset Repository | Primary Modality | Total Cells / Scope | Key Objective | Ground Truth |
| --- | --- | --- | --- | --- |
| **Tabula Sapiens / HCA** | scRNA-seq | ~500,000 cells | Taxonomic generalization | Expert-curated consensus |
| **CZ CELLxGENE** | scRNA-seq | >30 million cells | Platform robustness | Standardized pipelines |
| **TISCH2 / 3CA** | scRNA-seq (Oncology) | >2 million cells | Malignant discrimination | Curated marker mapping |
| **SCONE** | scRNA-seq | Controlled mixtures | Mutational heterogeneity | Known synthetic mixtures |

### Spatial Transcriptomic Requirements

* **High-Resolution Imaging Data:** Datasets with single-cell or sub-cellular resolution (e.g., MERFISH, Xenium, CosMx) where ground truth is established via paired single-nucleus sequencing or expert manual annotation.
* **Sequencing-Based Spatial Grids:** Lower-resolution datasets (e.g., Visium HD, Stereo-seq) that are the gold standard for benchmarking spatial layer clustering, anatomical region identification, and deconvolution accuracy.

| Spatial Modality | Platforms | Resolution | Benchmark Goal | Ground Truth |
| --- | --- | --- | --- | --- |
| **Imaging-Based ST** | Xenium, MERFISH, CosMx | Single-cell | Precise mapping | CODEX + expert segmentation |
| **Sequencing-Based ST** | Visium HD, Stereo-seq | Spot-level | Deconvolution | Semi-simulated aggregation |

## 4. Proposed Architecture: The Tri-Agent Ontology Framework

This proposed architecture simulates the rigorous peer-review process of a multidisciplinary team of computational biologists, utilizing three distinct, interacting personas.

### A. The Input & Information Hub

* **Inputs:** * *scRNA-seq:* Differentially expressed gene lists (ranked by log-fold change and statistical significance) and preprocessed AnnData objects.
* *Spatial Transcriptomics:* Gene scores, spatial coordinate matrices, and spatial neighborhood graphs.


* **Abstract Capabilities:** The environment provides the agents with modular functional capabilities:
1. Pathway enrichment analysis using GSEA.
2. Tool-calling APIs for marker gene sets from overall databases.
3. Mathematical similarity calculations (comparing cluster expression vectors to established reference embeddings).
4. Extendable infrastructure to other tools and databases.



### B. The Agentic Network

**1. Agent Alpha: The Molecular Analyst**

* **Role:** Analyzes the purely transcriptomic signals.
* **Action:** Ingests the top marker genes (e.g., top 50 DEGs), executes gene set enrichment analysis, and retrieves established literature markers. It outputs a probabilistic list of candidate cell types or functional states based *exclusively* on the gene expression profile.
* **Output:** A list of 3–5 candidate cell types or functional states with confidence scores.

**2. Agent Beta: The Spatial Contextualizer**

* **Role:** Analyzes the physical neighborhood and architectural logic.
* **Action:** Examines the spatial adjacency matrix utilizing *nichecards* and enforces the ASCT+B anatomical hierarchy. If Agent Alpha proposes an annotation that violates biological spatial coherence (e.g., proposing an isolated epithelial cell type within a dense region of distinct neural cells), Agent Beta applies spatial coherence penalties, flags the biological implausibility, and requests a re-evaluation.
* **Output:** A refined list of candidate cell types or functional states with adjusted confidence scores.

**3. Agent Gamma: The Ontologist & Critic**

* **Role:** The final decision-maker, standardizer, and arbitrator.
* **Action:** Mediates the analysis between the Molecular Analyst and the Spatial Contextualizer. It forces the final, agreed-upon annotation to strictly map to standardized biological nomenclature (e.g., Cell Ontology IDs) and enforces hierarchical probability propagation.
* **Output:** The final cell type, state, or activity, a calculated confidence score, and a concise, biologically grounded reasoning chain.

## 5. Evaluation Plan

To rigorously evaluate the agent, performance will be compared against gold-standard annotations derived from curated benchmark datasets across four primary axes:

### A. Accuracy Metrics

* **Cell Type Classification Accuracy:** Precision, recall, and F1-score for each cell type to identify states the agent struggles to distinguish. This includes advanced ontology-aware metrics like Hop-based HF1 and CyteOnto semantic similarity.
* **Spatial Coherence Score (SCS):** Measured via cosine similarity between the predicted cell type expression profile and the actual cell type expression profile, alongside metrics like the CHAOS score and Average Silhouette Width (ASW).
* **Ontological Consistency:** Measuring the degree to which annotations adhere to standardized biological ontologies, ensuring hierarchical correctness.

### B. Robustness Testing

* **Sensitivity Analysis:** Evaluating performance under varying conditions (input noise, batch effect sensitivity, varying marker lists, and differing spatial resolutions). This is particularly important for oncology datasets, which are notoriously noisy and batch-heavy.
* **Cross-Dataset Generalization:** Testing zero-shot cross-species generalization and the ability to adapt to new datasets from different experimental platforms (comparing against baselines like Harmony, fastMNN, Scanorama, and sysVI).

### C. Efficiency Analysis

* **Computational Performance:** Measuring inference latency and the resources required (e.g., token optimization using only the top 50 markers) to process datasets. Benchmarking will utilize tools like BixBench and CoLLAB.
* **Scalability:** Evaluating how performance and efficiency scale with increasing dataset size and complexity to assess real-world feasibility.

### D. Agent Evaluation

* **Human Evaluation:** Domain experts will review the agent's annotations, evaluating biological plausibility and reasoning logic.
* **Automated Evaluation:** Automated scripts will programmatically score annotations against the established ground truth.

## 6. Implementation Plan

### A. Technology Stack

* **Agent Framework:** LangChain
* **Large Language Models:** Gemini 3 / GPT-4o / Claude 3.5 Sonnet
* **Transcriptomics Analysis Tools:** Scanpy
* **Data Processing & Analytics:** Pandas, NumPy, SciPy
* **Parallelization:** concurrent.futures (ThreadPoolExecutor)
* **Visualization:** Matplotlib, Seaborn

### B. Interface Design

* **Option 1: Command Line Interface (CLI)**
* Accepts data paths (e.g., `.h5ad` files) and matrices for gene expression and clustering/factorization results.
* Outputs structured JSON results and generates visual plots (annotated UMAPs for single-cell data and annotated spatial maps).


* **Option 2: Web User Interface (UI)**
* A portal for dataset uploads.
* Generates an interactive HTML report visualizing annotated UMAPs and spatial maps.
* Exposes the agents' reasoning chains.
* Features an embedded chat interface for human-in-the-loop refinement and querying.


