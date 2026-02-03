# ChromBERT-tools: A versatile toolkit for context-specific embedding of transcription regulators across different cell types

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight toolkit designed to generate and interpret regulation-informed embeddings derived from ChromBERT, providing user-friendly command-line interfaces (CLIs) and Python APIs.
![ChromBERT-tools Framework](docs/_static/figure1.png "Framework")

---

## Installation
ChromBERT-tools depends on the ChromBERT environment (packages + pre-trained models files). The official Singularity image is the easiest option and it already includes ChromBERT-tools.

### 1) ChromBERT environment (recommended)
Install Apptainer and pull the image:
```bash
conda install -c conda-forge apptainer
apptainer pull chrombert.sif oras://docker.io/chenqianqian515/chrombert:20260202
```
Optional: download the image from the [Google Drive link](https://drive.google.com/file/d/14I-BQxrBNPwdZn-TKaG0Z8lpNiJlUd1f/view?usp=drive_link).

Quick check:
```bash
singularity exec /path/to/chrombert.sif python -c "import chrombert; print('hello chrombert')" # need some time
singularity exec /path/to/chrombert.sif chrombert-tools
```

### 2) ChromBERT pre-trained models files
Download models and annotations to `~/.cache/chrombert/data`:
```shell
singularity exec /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb
```
If Hugging Face is slow, add `--hf-endpoint <mirror>`:
```shell
singularity exec /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>
```

### 3) (optional) Update the Singularity image with the latest ChromBERT-tools
To add new packages or update existing ones, edit edit_image.def and rebuild a new image.
Here we update ChromBERT-tools as an example:

```bash
git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
cd ChromBERT-tools
apptainer build <new_image_name>.sif edit_image.def
```

## Usage

ChromBERT-tools supports two ways to run:

- **Command-line interface (CLI)** — run from a terminal (bash commands)
- **Python API** — call functions in Python code


## ChromBERT-tools CLI
For usage examples, see the Jupyter notebooks in [`examples/cli/`](examples/cli/).
you can run examples with Jupyter Notebook:
```bash
git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
cd ChromBERT-tools/examples/
singularity exec --nv /path/to/chrombert.sif jupyter-notebook # start Jupyter Notebook with GPU support
```

For detailed usage, please check the documentation: [chrombert-tools.readthedocs.io](https://chrombert-tools.readthedocs.io/en/latest/).
### Generation of regulation-informed embeddings
- [embed_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cistrome.html): Extract cistrome embeddings for specified regions  
- [embed_gene](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_gene.html): Extract gene embeddings  
- [embed_region](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_region.html): Extract region embeddings for specified regions  
- [embed_regulator](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_regulator.html): Extract regulator embeddings for specified regions
- [embed_cell_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_cistrome.html): Extract cell-type-specific cistrome embeddings for specified regions  
- [embed_cell_gene](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_gene.html): Extract cell-type-specific gene embeddings  
- [embed_cell_region](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_region.html): Extract cell-type-specific region embeddings for specified regions  
- [embed_cell_regulator](https://chrombert-tools.readthedocs.io/en/latest/commands/embed_cell_regulator.html): Extract cell-type-specific regulator embeddings for specified regions  


### Interpretation of regulation-informed embeddings
- [infer_ep](https://chrombert-tools.readthedocs.io/en/latest/commands/infer_ep.html): Infer enhancer-promoter loops
- [infer_regulator_network](https://chrombert-tools.readthedocs.io/en/latest/commands/infer_regulator_network.html): Infer regulator-regulator networks on specified regions  
- [infer_cell_key_regulator](https://chrombert-tools.readthedocs.io/en/latest/commands/infer_cell_key_regulator.html): Infer cell-type-specific key regulators
- [find_driver_in_transition](https://chrombert-tools.readthedocs.io/en/latest/commands/find_driver_in_transition.html): Find driver factors in cell-state transitions
- [find_context_specific_cofactor](https://chrombert-tools.readthedocs.io/en/latest/commands/find_context_specific_cofactor.html): Find context-specific cofactors in different regions  

### Cistrome imputation
- [impute_cistrome](https://chrombert-tools.readthedocs.io/en/latest/commands/impute_cistrome.html): Impute cistrome data on specified regions 


## ChromBERT-tools API

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

```python
from chrombert_tools import embed_gene, embed_region, embed_cistrome, embed_regulator, infer_ep, infer_regulator_network, impute_cistrome
```

For detailed usage examples, see the Jupyter notebooks in [`examples/api/`](examples/api/).

## Contact us
If you have any questions or suggestions, please feel free to contact us at [2211083@tongji.edu.cn](mailto:2211083@tongji.edu.cn).

