# ChromBERT-tools: Utilities for ChromBERT-based regulatory analysis

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight companion library that wraps these capabilities into easy-to-use command-line and Python utilities.

**ChromBERT-tools v1.0 will be released on December 26, 2025**

---

## What you can do with ChromBERT-tools

### 1. General representations

- **Region embeddings** – generate ChromBERT-based embeddings for genomic regions.
- **Gene embeddings** – generate embeddings for genes specified by gene symbols or IDs.  
- **Regulator embeddings** – generate embeddings for transcription factors / regulators.  
- **Cistrome embeddings** – generate embeddings for cistomes.

### 2. Cell- and condition-specific representations

- **Cell-specific embeddings** – obtain region, gene and regulator embeddings adapted to a given cell type or condition.  
- **Cell-specific TRNs** – infer transcriptional regulatory networks over user-specified regions.

### 3. Cistrome imputation

- **Impute missing cistromes** – predict cistromes for regulators or contexts that lack direct ChIP-like assays.

### 4. Dynamic analysis and driver regulator discovery

- **Cell-state transitions** – rank putative driver regulators underlying specific transitions between cell types or conditions.  
- **Functional region contrasts** – identify regulators whose context-specific embeddings best discriminate between two sets of functional regions.