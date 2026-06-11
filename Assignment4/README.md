# Assignment 4 – Generative Models with GAN

Ben-Gurion University, Deep Learning Course

## Overview

Implements a standard GAN and Conditional GAN (cGAN) on the Adult Census Income tabular
dataset, covering:

- **Section 6** – Main experiments: GAN + cGAN training across 3 seeds, detection and efficacy evaluation
- **Section 7** – Discrete features: Softmax vs Gumbel-Softmax with straight-through estimator
- **Section 8** – Mode collapse: induce (5:1 D:G updates), diagnose (coverage ratio), mitigate (minibatch std)
- **Section 9** – Open-ended: Spectral normalization on the discriminator

## Files

```
Assignment4/
├── gan_assignment.py       # Source: jupytext percent-format Python file
├── gan_assignment.ipynb    # Notebook generated from the .py file
├── adult.arff              # Adult Census Income dataset (ARFF format, included)
├── figures/                # All plots saved here during execution
└── README.md               # This file
```

The `.py` file is the single source of truth. The `.ipynb` is kept in sync via jupytext.
**Always edit `gan_assignment.py`; re-sync the notebook with the command below.**

## Environment Setup

```bash
cd Assignment4
conda env create -f conda_requirement.yaml
conda activate DL_A4_env
```

## Running as a Notebook

```bash
cd Assignment4

# Open in JupyterLab / Jupyter Notebook
jupyter notebook gan_assignment.ipynb
# or
jupyter lab gan_assignment.ipynb
```

Run all cells top-to-bottom (Kernel → Restart & Run All).

The notebook will:
1. Load `adult.arff` (must be present in `Assignment4/`).
2. Train GAN + cGAN for each of 3 seeds (~200 epochs each).
3. Run Sections 7, 8, 9 experiments.
4. Save all figures to `figures/`.

Expected runtime: ~15 min on GPU (CUDA), ~60–90 min on CPU.

## Running as a Plain Python Script

```bash
cd Assignment4
conda activate DL_A4_env
python gan_assignment.py
```

## Syncing .py ↔ .ipynb

After editing `gan_assignment.py`:

```bash
# Regenerate notebook (overwrites outputs)
jupytext --to notebook --set-kernel python3 gan_assignment.py

# Or update in place (preserves existing cell outputs)
jupytext --update --to notebook gan_assignment.ipynb
```

After running cells in the notebook and wanting to sync back to `.py`:

```bash
jupytext --to py:percent gan_assignment.ipynb
```

## Output

All figures are written to `figures/`:

| File | Content |
|------|---------|
| `eda_class_dist.png` | Income class distribution (EDA) |
| `eda_continuous.png` | Continuous feature distributions (EDA) |
| `eda_categorical.png` | Categorical feature distributions (EDA) |
| `gan_loss_seed42.png` | GAN training loss curves (seed 42) |
| `gan_loss_seed123.png` | GAN training loss curves (seed 123) |
| `gan_loss_seed456.png` | GAN training loss curves (seed 456) |
| `cgan_loss_seed42.png` | cGAN training loss curves (seed 42) |
| `cgan_loss_seed123.png` | cGAN training loss curves (seed 123) |
| `cgan_loss_seed456.png` | cGAN training loss curves (seed 456) |
| `gan_dist_continuous.png` | Continuous feature histograms: GAN |
| `gan_dist_categorical.png` | Categorical bar charts: GAN |
| `cgan_dist_continuous.png` | Continuous feature histograms: cGAN |
| `cgan_dist_categorical.png` | Categorical bar charts: cGAN |
| `gan_corr.png` / `cgan_corr.png` | Correlation matrix comparison |
| `s7_loss_comparison.png` | Softmax vs Gumbel-ST loss curves |
| `s7_entropy.png` | Per-feature entropy comparison |
| `s8_collapse_loss.png` | Collapse loss curves (baseline vs 5:1) |
| `s8_coverage.png` | Coverage ratio bar chart |
| `s8_all_loss.png` | All three collapse experiments |
| `s9_sn_loss.png` | Spectral-norm vs baseline loss |

## Dataset

The Adult (Census Income) dataset must be present as `adult.arff` in the `Assignment4/`
directory before running. The file is included in the submission.
