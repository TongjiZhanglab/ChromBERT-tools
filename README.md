# ChromBERT-tools: Utilities for ChromBERT-based regulatory analysis

*Turn ChromBERT into practical workflows for regulatory genomics.*

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight companion library that wraps these capabilities into easy-to-use command-line and Python utilities.

ChromBERT-tools helps you:

- compute ChromBERT-based representations of genomic regions, genes and regulators;
- infer transcriptional regulatory networks (TRNs) in general and cell-specific settings;
- impute missing cistromes for regulators or contexts without direct ChIP-like assays;
- compare regulatory programs between cell types, conditions or functional region sets and identify candidate driver regulators.

---

## What you can do with ChromBERT-tools

### 1. General representations (pre-trained ChromBERT)

- **Region embeddings** – generate ChromBERT-based embeddings for genomic regions (e.g. from BED files).  
- **Gene embeddings** – generate embeddings for genes specified by gene symbols or IDs (via their TSS regions).  
- **Regulator embeddings** – generate embeddings for transcription factors / regulators.  
- **Cistrome embeddings** – generate embeddings summarizing cistromes defined by TFs, marks or cell types.

### 2. Cell- and condition-specific representations (fine-tuned ChromBERT)

- **Cell-specific embeddings** – obtain region, gene and regulator embeddings adapted to a given cell type or condition via fast fine-tuning of ChromBERT.  
- **Cell-specific TRNs** – infer transcriptional regulatory networks over user-specified regions or loci using cell-type-specific chromatin or expression data.

### 3. Cistrome imputation

- **Impute missing cistromes** – predict cistromes for regulators or contexts that lack direct ChIP-like assays using ChromBERT-based models.

### 4. Dynamic analysis and driver regulator discovery

- **Cell-state transitions** – rank putative driver regulators underlying specific transitions between cell types or conditions.  
- **Functional region contrasts** – identify regulators whose context-specific embeddings best discriminate between two sets of functional regions.