# ChromBERT-tools: Command-line tools for ChromBERT-based regulatory analysis
**ChromBERT-tools v1.0 will be released on December 26, 2025**

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight GitHub toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI).


---

## Installation
ChromBERT-tools is a lightweight GitHub toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI). You need to install the ChromBERT environment including ChromBERT dependencies and datasets.

### Installing ChromBERT Dependencies
If you have already installed ChromBERT dependencies, you can skip this step and proceed to [Installing ChromBERT-tools](#installing-chrombert-tools).

For direct use of these CLI tools, it is recommended to utilize the ChromBERT [Singularity image](https://drive.google.com/file/d/10Mma4jZsloFP2EMFuXEWXNH5iPHCIn9H/view?usp=drive_link). **This image includes almost all packages needed by ChromBERT and ChromBERT-tools**, including flash-attention-2, transformers, pytorch, etc.

**If you want to install from source and use development mode, you can follow the instructions in the [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT) repository.**

To use the Singularity image, you need to install `Apptainer` first:
```bash
conda install -c conda-forge apptainer
apptainer pull chrombert.sif oras://docker.io/chenqianqian515/chrombert:20251225
```
Alternatives (optional download methods):
- Download the Singularity image directly from the [Google Drive link](https://drive.google.com/file/d/10Mma4jZsloFP2EMFuXEWXNH5iPHCIn9H/view?usp=drive_link) and save it as chrombert.sif.
- Oras pull
    ```bash
    conda install -c conda-forge oras
    oras pull registry-1.docker.io/chenqianqian515/chrombert:20251225 -o . && mv chrombert_20251225.sif chrombert.sif
    ```

Then you can test whether it was successfully installed:
```bash
singularity exec --nv /path/to/chrombert.sif python -c "import chrombert; print('hello chrombert')"
singularity exec --nv /path/to/chrombert.sif chrombert-tools
```

### Installing ChromBERT Dataset
Download the required pre-trained model and annotation data files from Hugging Face to `~/.cache/chrombert/data`.
You can download hg38 (200bp, 1kb, 2kb, 4kb resolution datasets) and mm10 (1kb resolution dataset):
```shell
singularity exec --nv /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb
```

Alternatively, if you're experiencing significant connectivity issues with Hugging Face, you can use the `--hf-endpoint` option to connect to an available mirror:
```shell
singularity exec --nv /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>
```

### Installing ChromBERT-tools
```bash
git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
cd ChromBERT-tools
pip install -e .
```
To verify the installation, execute the following command:
```bash
chrombert-tools
```

## Usage

ChromBERT-tools supports two ways to run:

- **Command-line interface (CLI)** — run from a terminal (bash commands)
- **Python API** — call functions in Python code


## ChromBERT-tools CLI
For detailed usage, please check the documentation: [chrombert-tools.readthedocs.io](https://chrombert-tools.readthedocs.io/en/latest/).

For detailed usage examples, see the Jupyter notebooks in [`examples/cli/`](examples/cli/).

### General (pre-trained)
- [embed_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cistrome.html): Extract cistrome embeddings for specified regions  
- [embed_gene](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_gene.html): Extract gene embeddings  
- [embed_region](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_region.html): Extract region embeddings for specified regions  
- [embed_regulator](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_regulator.html): Extract regulator embeddings for specified regions  
- [infer_trn](https://chrombert-tools.readthedocs.io/en/latest/commands/infer_trn.html): Infer transcriptional regulatory networks (TRNs) on specified regions  

### Cell-type-specific
- [infer_cell_trn](https://chrombert-tools.readthedocs.io/en/latest/commands/infer_cell_trn.html): Infer cell-type-specific TRNs on specified regions and key regulators  
- [embed_cell_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_cistrome.html): Extract cell-type-specific cistrome embeddings for specified regions  
- [embed_cell_gene](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_gene.html): Extract cell-type-specific gene embeddings  
- [embed_cell_region](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_region.html): Extract cell-type-specific region embeddings for specified regions  
- [embed_cell_regulator](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_regulator.html): Extract cell-type-specific regulator embeddings for specified regions  

### Cistrome imputation
- [impute_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/impute_cistrome.html): Impute cistrome data on specified regions  

### Driver analysis
- [find_driver_in_dual_region](https://chrombert-tools.readthedocs.io/en/latest/commands/find_driver_in_dual_region.html): Find driver factors in dual-functional regions  
- [find_driver_in_transition](https://chrombert-tools.readthedocs.io/en/latest/commands/find_driver_in_transition.html): Find driver factors in cell-state transitions  


## ChromBERT-tools API

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

```python
from chrombert_tools import embed_gene, embed_region, embed_cistrome, embed_regulator, infer_trn, impute_cistrome
```

For detailed usage examples, see the Jupyter notebooks in [`examples/api/`](examples/api/).

## Contact us
If you have any questions or suggestions, please feel free to contact us at [2211083@tongji.edu.cn](mailto:2211083@tongji.edu.cn).

