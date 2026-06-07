# Assignment 3 — Lyrics Generation using RNNs

Ben-Gurion University of the Negev — Faculty of Computer & Information Science

## Setup

This assignment uses a **conda** environment (not pip/venv).

### 1. Create the environment (first time only)

```bash
cd Assignment3
conda env create -f conda_requirement.yaml
```

This installs Python 3.11, PyTorch (CPU), JupyterLab, jupytext, and all other dependencies into an env named `DL_A3_env`.

### 2. Activate the environment

```bash
conda activate DL_A3_env
```

### 3. Launch Jupyter

```bash
jupyter lab
```

Open `src/assignment3_lyrics_generation.ipynb`.

---

## Notebook ↔ Python file sync (jupytext)

The `src/` folder contains a paired `.ipynb` + `.py` (percent format). They are kept in sync with jupytext:

```bash
cd Assignment3
conda run -n DL_A3_env jupytext --sync ./src/assignment3_lyrics_generation.py
```

Edit either file; run `--sync` to propagate changes to the other.

---

## Data

- `data/lyrics_train_set.csv` / `data/lyrics_test_set.csv` — lyrics dataset
- `data/midi_files/` — MIDI files for bonus analysis
- `results/` — generated figures and statistics
