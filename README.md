# ChromBERT-tools: Utilities for ChromBERT-based regulatory analysis

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight GitHub toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI).

**ChromBERT-tools v1.0 will be released on December 26, 2025**

---

## Installation
ChromBERT-tools is a lightweight GitHub toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI). You need to install the ChromBERT environment including ChromBERT dependencies and datasets.

### Installing ChromBERT Dependencies
If you have already installed ChromBERT dependencies, you can skip this step and proceed to [Installing ChromBERT-tools](#installing-chrombert-tools).

For direct use of these CLI tools, it is recommended to utilize the ChromBERT [Singularity image](). **These images include almost all packages needed by ChromBERT and ChromBERT-tools**, including flash-attention-2, transformers, pytorch, etc.

If you want to install from source and use development mode, you can follow the instructions in the [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT) repository.

To use the Singularity image, you need to install `singularity` (or `Apptainer`) first:
```bash
conda install -c conda-forge apptainer
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
chrombert_prepare_env --genome hg38 --resolution 1kb
```

Alternatively, if you're experiencing significant connectivity issues with Hugging Face, you can use the `--hf-endpoint` option to connect to an available mirror:
```shell
chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>
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
For detailed information on usage, please check out the documentation at [chrombert-tools.readthedocs.io](https://chrombert-tools.readthedocs.io/en/latest/).
