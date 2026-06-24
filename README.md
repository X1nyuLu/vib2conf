# Vib2Conf: AI-driven Discrimination of Molecular Conformations from Vibrational Spectra

[![arXiv](https://img.shields.io/badge/arXiv-2604.24310-c72c2c.svg)](https://arxiv.org/abs/2604.24310)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-vib2conf-dd9029)](https://huggingface.co/xinyulu/vib2conf)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Abstract

Retrieving or generating two-dimensional molecular structures from vibrational spectra has been well demonstrated via deep learning. However, deciphering three-dimensional molecular conformations remains challenging, primarily due to spectral ambiguities arising from conformational heterogeneity. To address this limitation, we propose **Vib2Conf**, a deep learning model that directly discriminates 3D molecular conformations from vibrational spectra. Specifically, we introduce an *attentional resampler* to distill conformation-sensitive features from sparse spectral signals, and incorporate a *Mixture-of-Experts (MoE)* module to partition the conformational space for precise geometric mapping. Together, these components enable Vib2Conf to achieve state-of-the-art top-1 recall exceeding **95%** on conventional spectrum–structure benchmarks, including QM9S, VB-Mols, and QMe14S. More importantly, Vib2Conf successfully discriminates near-isomeric conformers with a top-1 recall of **82.06%** on the VB-Confs test set, where conformational isomers differ by a root-mean-square deviation (RMSD) of only ~1 Å. Overall, Vib2Conf provides a promising approach for fine-grained spectrum-to-conformation analysis.

## Framework

<p align="center">
  <img src="./doc/toc.png" alt="Framework of Vib2Conf" width="90%"/>
</p>

## Installation

**Prerequisites:** Python 3.10+ and PyTorch 2.x with CUDA.

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install PyG extensions (match your PyTorch/CUDA version)
#    Reference: https://pytorch-geometric.readthedocs.io/en/latest/install/extension.html
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv pyg_lib \
    -f https://data.pyg.org/whl/torch-<TORCH_VERSION>+cu<CUDA_VERSION>.html
```

## Datasets & Checkpoints

All datasets and pretrained checkpoints are hosted on HuggingFace. Download them with:

```bash
pip install huggingface_hub
huggingface-cli download xinyulu/vib2conf --local-dir ./ \
    --exclude "README.md" ".gitattributes"
```

This will populate the `datasets/` and `checkpoints/` directories under the project root.

> 💡 **Tip for users in mainland China:** Prepend `HF_ENDPOINT=https://hf-mirror.com` to the command for faster downloads.

## Usage

### Training

We provide Vib2Conf along with two reimplemented baselines (SMEN and VibraCLIP). Representative commands are listed below.

```bash
# Vib2Conf — single-modality (Raman-only)
python main.py -train \
    --launch base \
    --model vib2conf_equiformer_moe_balance0001 \
    --ds qm9s \
    --task raman

# Vib2Conf — dual-modality (Raman + IR, multi-GPU)
torchrun --nproc_per_node=4 main.py -train --ddp \
    --launch base \
    --model vib2conf_equiformer_moe_concat_balance0001 \
    --ds vb_confs \
    --task raman-ir \
    --use_ema

# SMEN baseline (multi-GPU)
torchrun --nproc_per_node=2 main.py -train --ddp \
    --launch base \
    --config smen \
    --model smen \
    --ds qm9s \
    --task raman \
    --find-unused-parameters

# VibraCLIP baseline — single-modality
python main.py -train \
    --launch base \
    --config vibraclip \
    --model vibraclip \
    --ds qm9s \
    --task raman

# VibraCLIP baseline — dual-modality
python main.py -train \
    --launch base \
    --config vibraclip \
    --model vibraclip_dual \
    --ds qme14s \
    --task raman-ir
```

**Key arguments:**

| Argument    | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `--model`   | Model architecture (see [Available Models](#available-models))              |
| `--ds`      | Dataset name: `qm9s`, `qme14s`, `vb_confs`, or `vb_mols`                    |
| `--task`    | Spectral modality: `raman`, `ir`, or `raman-ir` (dual-modality)             |
| `--launch`  | Training preset: `base`, `smen`, or `vibraclip`                             |
| `--config`  | Config preset in `config.yaml`: `base`, `smen`, or `vibraclip`              |
| `--use_ema` | Enable Exponential Moving Average for model weights                         |

All hyperparameters are defined in `config.yaml` and can be overridden via command-line flags (e.g., `--batch-size`, `--epoch`, `--lr`).

### Evaluation

```bash
python eval.py \
    --ckpt checkpoints/vb_confs/raman/vib2conf_equiformer_moe_balance0001/<run>/epoch147.pth \
    --save results.pickle   # optional: export a DataFrame with SMILES-level analysis
```

The evaluation script reports Recall@1/3/5 for both spectrum→molecule and molecule→spectrum retrieval directions.

### Visualization

TensorBoard logs are saved under `runs/`, where you can inspect the complete ablation study records. Figures presented in the paper can be reproduced via `figures.ipynb`.

```bash
tensorboard --logdir runs/
```

## Available Models

We release the model variants used in our ablation studies, together with two existing baselines reimplemented under the Vib2Conf framework for fair comparison:

| Model                                          | Description                                                |
|------------------------------------------------|------------------------------------------------------------|
| `vib2conf_equiformer_moe_balance0001`          | **Best single-modality** model (Raman-only or IR-only)     |
| `vib2conf_equiformer_moe_concat_balance0001`   | **Best dual-modality** model (Raman + IR)                  |
| `vib2conf_equiformer_base_*`                   | Standard Equiformer variants (no MoE)                      |
| `vib2conf_equiformer_moe_*`                    | MoE Equiformer variants with different expert/pooling configs |
| `smen`                                         | EGNN + ViT with contrastive learning                       |
| `vibraclip` / `vibraclip_dual`                 | DimeNet++ + MLP with contrastive learning                  |

## Datasets

| Dataset    | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `qm9s`     | QM9S benchmark                                                           |
| `qme14s`   | QMe14S benchmark                                                         |
| `vb_mols`  | VB-Mols benchmark                                                        |
| `vb_confs` | VB-Confs: near-isomeric conformer discrimination (RMSD ~1 Å)             |

## Citation

If you find Vib2Conf useful in your research, please cite:

```bibtex
@article{lu2026vib2conf,
    title   = {Vib2Conf: AI-driven discrimination of molecular conformations from vibrational spectra}, 
    author  = {Xin-Yu Lu and De-Yi Lin and Tong Zhu and Bin Ren and Hao Ma and Guo-Kun Liu},
    year    = {2026},
    eprint  = {2604.24310},
    archivePrefix = {arXiv},
    primaryClass  = {physics.chem-ph},
    url     = {https://arxiv.org/abs/2604.24310}
}
```

## Acknowledgements

This repository is built upon our previous work [Vib2Mol](https://github.com/X1nyuLu/vib2mol).

## Contact

For questions or suggestions, please contact [xinyulu@stu.xmu.edu.cn](mailto:xinyulu@stu.xmu.edu.cn) or [open an issue](../../issues).