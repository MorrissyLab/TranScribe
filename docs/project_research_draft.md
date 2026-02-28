# TranScribe: An Agentic Framework for Automated Annotation of Single-cell and Spatial Transcriptomics

## 1. Introduction

The unprecedented advancement of highly multiplexed transcriptomic technologies has fundamentally transformed the landscape of modern biology, offering an unparalleled resolution into cellular heterogeneity, developmental trajectories, and complex tissue architectures. Single-cell RNA sequencing (scRNA-seq) has enabled the genome-wide profiling of millions of individual cells, laying the groundwork for massive, pan-organ reference atlases. Simultaneously, the rapid emergence of spatially resolved transcriptomics (ST) has introduced a critical physical dimension to this data, allowing researchers to study not only what genes are expressed, but precisely where these cells reside within intact tissue microenvironments. This spatial context is paramount for understanding cell-cell communication, functional tissue units, and the pathological restructuring of microenvironments in diseases such as cancer and neurodegeneration.  

However, despite these technological leaps in data generation, the computational interpretation of this data—specifically the accurate, reproducible annotation of cell types and spatial niches—remains a profound bottleneck. Traditional analytical pipelines rely heavily on deterministic clustering algorithms followed by heuristic, manual inspection of differentially expressed genes (DEGs). This process is notoriously subjective, deeply dependent on domain-specific human expertise, and highly susceptible to batch effects and platform-specific noise. While early supervised machine learning classifiers and deep learning models sought to automate this step, they frequently suffer from an inability to generalize across distinct biological conditions or differing sequencing protocols. Furthermore, they largely operate as "black boxes," providing raw classifications without transparent biological reasoning.  

The recent explosion of Large Language Models (LLMs) has introduced a promising new paradigm: the application of artificial intelligence capable of natural language reasoning, tool utilization, and autonomous logical deduction. Yet, deploying standalone, single-agent LLMs directly into bioinformatics pipelines presents significant risks. Base language models are prone to hallucinating marker associations, failing to grasp complex spatial geometries, and producing hyperconfident misclassifications when confronted with highly specialized or noisy biological data.  

To resolve these systemic challenges, the field of computational biology is rapidly pivoting toward multi-agent, collaborative reasoning frameworks. This comprehensive report outlines an exhaustive synthesis of current literature, curates a rigorous benchmarking dataset ecosystem, and proposes the architecture for **TranScribe**: a novel Tri-Agent Ontology Framework explicitly engineered to automate single-cell and spatial transcriptomic annotation. By simulating the peer-review dynamics of a multidisciplinary scientific panel, seamlessly integrating dynamic knowledge retrieval, and enforcing strict ontological and spatial coherence, TranScribe establishes a highly scalable, interpretable, and biologically faithful pipeline for next-generation transcriptomic analysis.

---

## 2. Literature Review

The transition from isolated computational models to multi-agent reasoning networks represents the cutting edge of biological data interpretation. A thorough synthesis of contemporary computational biology literature reveals three foundational pillars driving the evolution of transcriptomic annotation:

- Multi-agent consensus mechanisms  
- Dynamic knowledge-retrieval integration  
- Spatial-aware reasoning pipelines  

### Multi-Agent Consensus Mechanisms

The initial applications of LLMs in cell type annotation operated primarily in a zero-shot, single-agent capacity, querying models directly with lists of highly variable genes. While partially effective for canonical cell lineages, this approach frequently faltered when analyzing rare populations or transitional cellular states, often resulting in biologically inconsistent interpretations due to a fundamental lack of internal verification.  

To mitigate these inherent vulnerabilities, contemporary research has aggressively pivoted toward multi-agent architectures that simulate the rigorous, iterative peer-review processes characteristic of multidisciplinary scientific panels.

Systems such as **CASSIA** and **CellAgent** have pioneered the strategic division of analytical labor among specialized, interacting LLM personas. CellAgent conceptualizes its architecture through distinct Planner, Executor, and Evaluator roles to automate end-to-end scRNA-seq analysis, ranging from preprocessing and batch correction to final cell-type annotation. CASSIA deploys a multi-agent AI network explicitly designed to guard against hallucinations and calibrate confidence. By explicitly requiring agents to produce transparent reasoning chains and subject their predictions to internal quality assessments, CASSIA demonstrates significantly improved annotation accuracy on benchmark datasets, particularly concerning complex and rare cell populations.  

Similarly, the **CyteType** framework introduces a sophisticated multi-agent ecosystem utilizing hypothesis-driven reasoning. In this framework, an Annotator agent resolves marker gene ambiguity by generating multiple competing hypotheses regarding cell identity. These hypotheses are systematically tested against full expression profiles, pathway enrichment data, and broader study contexts. Crucially, CyteType introduces a Reviewer agent that simulates an expert panel, performing automated reference checking against marker databases and assigning confidence scores. If the evidence is insufficient or contradictory, the Reviewer triggers a re-annotation cycle, transforming the annotation process from a static classification task into an evidence-based, iterative characterization loop.  

In the context of spatial transcriptomics, frameworks like **SpatialAgent** and **NicheAgent** employ an "Analyst–Consensus–Reviewer" triad to manage the complexities of microenvironmental mapping. Independent Analyst agents evaluate localized transcriptomic signals, while a Consensus module fuses these independent proposals. The Reviewer role is explicitly tasked with enforcing spatial coherence, actively preventing phenomena such as "label collapse"—where a model erroneously propagates a single dominant cell type across distinct anatomical boundaries—ensuring the final annotation aligns logically with the surrounding tissue structure.

---

### Knowledge-Retrieval Integration

A critical vulnerability of standalone LLMs is their reliance on static internal weights. This internal knowledge frequently misaligns with rapidly evolving scientific consensus, localized study contexts, or niche biological domains.  

State-of-the-art agentic frameworks resolve this limitation through Retrieval-Augmented Generation (RAG) and active tool-calling, embedding external, structured biological databases directly into the analytical reasoning loop.

Biomedical ontologies—formal frameworks providing computable representations of domain knowledge—serve as foundational scaffolds. Resources such as:

- Gene Ontology (GO)  
- Human Phenotype Ontology (HPO)  
- Cell Ontology (CL)  

enable semantic interoperability and structured reasoning.

Frameworks like **Biomni** and **BioMaster** equip LLM agents with the ability to execute API calls to external databases (e.g., NCBI, Reactome, PubMed), cross-referencing proposed cell types against literature markers and enrichment algorithms.

CyteType exemplifies deep integration by constructing an internal pseudo-bulk expression database. Rather than evaluating only top DEGs, agents query full expression profiles before finalizing annotations. Frameworks like **KGARevion** further integrate knowledge graphs, enabling structured semantic triplet reasoning within biomedical domains.

---

### Spatial-Aware Reasoning Pipelines

Translating tissue architecture into a format digestible by language models presents a fundamental computational challenge. Standard LLMs exhibit limited geometric reasoning capabilities.

Two dominant approaches have emerged:

1. **Latent spatial embedding integration**  
   Frameworks like **SPELL (Spatial Prompt-Enhanced Zero-Shot Learning)** use Graph Autoencoders to compress k-nearest neighbor graphs into latent embeddings. Removing spatial context from prompts leads to dramatic performance degradation, demonstrating spatial awareness is essential.

2. **Text-based neighborhood encoding**  
   Lightweight frameworks such as NicheAgent convert coordinates into structured semantic representations termed *nichecards*. A nichecard encodes:
   - Canonical marker genes  
   - Expression centroids  
   - Neighboring cell-type frequencies within a defined radius  

This structured representation allows LLMs to enforce deterministic spatial coherence without processing raw coordinate matrices.

---

## 3. Benchmark Dataset Curation

Robust evaluation requires expert-curated ground-truth datasets. The benchmarking ecosystem spans both:

- Comprehensive multi-organ reference atlases  
- Complex disease microenvironments  

across single-cell and spatial modalities.

### Single-Cell Transcriptomic Requirements

#### Comprehensive Reference Atlases

- Human Cell Landscape  
- Tabula Sapiens  
- Mouse Cell Atlas (MCA)  
- CZ CELLxGENE Discover  

These repositories encompass tens of millions of cells across diverse tissues and hierarchical annotations.

| Dataset Repository | Primary Modality | Total Cells / Scope | Key Objective | Ground Truth |
|-------------------|------------------|--------------------|---------------|-------------|
| Tabula Sapiens / HCA | scRNA-seq | ~500,000 cells | Taxonomic generalization | Expert-curated consensus |
| CZ CELLxGENE | scRNA-seq | >30 million cells | Platform robustness | Standardized pipelines |
| TISCH2 / 3CA | scRNA-seq (Oncology) | >2 million cells | Malignant discrimination | Curated marker mapping |
| SCONE | scRNA-seq | Controlled mixtures | Mutational heterogeneity | Known synthetic mixtures |

---

### Spatial Transcriptomic Requirements

#### Imaging-Based Platforms

- MERFISH  
- Xenium In Situ  
- CosMx SMI  

Single-cell/subcellular resolution paired with CODEX protein mapping and matched snRNA-seq for triangulated ground truth.

#### Sequencing-Based Platforms

- Visium HD  
- Stereo-seq  

Spot-level resolution requiring deconvolution benchmarking via semi-simulated aggregated datasets.

| Spatial Modality | Platforms | Resolution | Benchmark Goal | Ground Truth |
|------------------|----------|-----------|---------------|--------------|
| Imaging-Based ST | Xenium, MERFISH, CosMx | Single-cell | Precise mapping | CODEX + expert segmentation |
| Sequencing-Based ST | Visium HD, Stereo-seq | Spot-level | Deconvolution | Semi-simulated aggregation |

---

## 4. Proposed Architecture: The Tri-Agent Ontology Framework

TranScribe simulates a multidisciplinary research team via:

- **Agent Alpha** – Molecular Analyst  
- **Agent Beta** – Spatial Contextualizer  
- **Agent Gamma** – Ontologist & Critic  

### A. Input & Information Hub

Provides:

- Preprocessed AnnData objects  
- DEG rankings  
- Spatial neighbor graphs  
- Tool-calling APIs (GSEA, marker DB querying, similarity metrics)  
- Modular extensibility  

---

### B. Agent Roles

#### 1. Agent Alpha – Molecular Analyst

- Consumes top 50 DEGs  
- Executes enrichment and marker validation  
- Outputs 3–5 candidate identities with confidence scores  

#### 2. Agent Beta – Spatial Contextualizer

- Uses nichecards  
- Enforces ASCT+B anatomical hierarchy  
- Applies spatial coherence penalties/bonuses  
- Outputs refined candidate list  

#### 3. Agent Gamma – Ontologist & Critic

- Arbitrates Alpha/Beta debate  
- Maps outputs to Cell Ontology IDs  
- Enforces hierarchical probability propagation  
- Produces final standardized label + reasoning chain  

---

## 5. Evaluation Plan

### A. Accuracy Metrics

- Precision, Recall, F1  
- Hop-based HF1 (ontology-aware metric)  
- CyteOnto semantic similarity  
- CHAOS score  
- Average Silhouette Width (ASW)  
- Spatial Coherence Score (SCS)  
- Ontological Consistency Score  

---

### B. Robustness Testing

- Batch effect sensitivity  
- Synthetic noise injection  
- Comparison to Harmony, fastMNN, Scanorama, sysVI  
- Zero-shot cross-species generalization  

---

### C. Efficiency & Scalability

- Token optimization (top 50 markers)  
- Inference latency measurement  
- Benchmarking via BixBench and CoLLAB  

---

## 6. Implementation Plan

### A. Technology Stack

- **LangChain** – agent orchestration  
- **Gemini 3 / GPT-4o / Claude 3.5 Sonnet** – LLM backend  
- **Scanpy** – transcriptomics preprocessing  
- **Pandas / NumPy / SciPy** – analytics  
- **concurrent.futures (ThreadPoolExecutor)** – parallelization  
- **Matplotlib / Seaborn** – visualization  

---

### B. Interface Design

#### Command Line Interface (CLI)

- Accepts `.h5ad` files  
- Outputs structured JSON  
- Generates annotated UMAP and spatial maps  

#### Web UI

- Dataset upload portal  
- Interactive HTML report  
- Exposed reasoning chains  
- Embedded chat for human-in-the-loop refinement  

