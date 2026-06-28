# AGResU-Net — BraTS MRI Brain Tumor Segmentation

> Reproduction and enhancement of *Attention Gate ResU-Net for Automatic MRI Brain Tumor Segmentation* (Zhang et al., 2020) using the BraTS 2017, 2018, and 2019 datasets.

**Team:** Layan Abandeh · Mira Diab · Sewar Abdelrahman  
**Course:** Machine Learning — Dr. Serin Atiani

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup & Requirements](#setup--requirements)
- [Usage](#usage)
- [Results](#results)
- [Enhancements](#enhancements)
- [Reports](#reports)
- [Reference](#reference)

---

## Overview

This project implements **AGResU-Net**, a 2D encoder-decoder network that combines:
- **Residual blocks** in both encoder and decoder paths
- **Attention gates** on skip connections to focus on relevant tumor regions
- A combined **Generalized Dice Loss + Weighted Cross Entropy** loss function

We first reproduced the paper's pipeline (baseline), then applied a series of targeted enhancements to push performance closer to — and in some cases beyond — the reported benchmarks.

---

## Architecture

```
Input (128×128×4 MRI)
        │
   Encoder 1 ── Residual Block ── 64 filters  ── 128×128
        │
   Encoder 2 ── Residual Block ── 128 filters ──  64×64
        │
   Encoder 3 ── Residual Block ── 256 filters ──  32×32
        │
     Bridge ─── Residual Block ── 512 filters ──  16×16
        │
   Decoder 3 ── Upsample + Attention Gate + Residual ── 256 ── 32×32
        │
   Decoder 2 ── Upsample + Attention Gate + Residual ── 128 ── 64×64
        │
   Decoder 1 ── Upsample + Attention Gate + Residual ──  64 ── 128×128
        │
   Output ────── Conv 1×1 + Softmax ── 4 classes ── 128×128
```

**Input:** 4-channel MRI (FLAIR, T1, T1ce, T2)  
**Output:** 4-class segmentation mask (Background, Necrosis/NCE, Edema, Enhancing Tumor)

---

## Repository Structure

```
AGResU-Net-BraTS/
│
├── baseline.ipynb          # Phase 3: pipeline reproduction (2-epoch verification run)
├── enhanced.ipynb          # Final model: full 40-epoch training with all enhancements
│
├── reports/
│   ├── phase3_report.pdf           # Phase 3 technical report
│   └── enhancement_report.docx    # Enhancement summary and final results
│
└── README.md
```

---

## Setup & Requirements

### Prerequisites

- Python 3.8+
- TensorFlow 2.x (GPU recommended)
- CUDA-compatible GPU (8 GB+ VRAM recommended)

### Install dependencies

```bash
pip install tensorflow numpy nibabel scikit-image matplotlib scipy
```

### Download the datasets

Download BraTS 2017, 2018, and 2019 via Kaggle and place them as follows:

```
data/
├── brats17/
├── brats2018/
└── brats2019/
```

Each patient folder must contain all four MRI modalities (FLAIR, T1, T1ce, T2) plus the segmentation mask.

---

## Usage

### Baseline (Phase 3 pipeline verification)

Open `baseline.ipynb` and run all cells. This is a short 2-epoch run intended to verify the end-to-end pipeline — GPU detection, data loading, model build, and plot generation. It does **not** produce competitive Dice scores.

### Enhanced (Final model)

Open `enhanced.ipynb` and run all cells. This runs the full 40-epoch training with all enhancements applied. Expects the BraTS datasets under `data/` as described above.

> **Tip:** Run `baseline.ipynb` first to confirm your environment is set up correctly before launching the full training run.

---

## Results

### Soft Dice Scores

| Dataset | WT | TC | ET |
|---|---|---|---|
| BraTS 2017 | **0.890** | **0.795** | **0.836** |
| BraTS 2018 | **0.884** | **0.813** | **0.821** |
| BraTS 2019 | **0.892** | **0.842** | **0.827** |
| Paper target (BraTS 2017) | 0.880 | 0.781 | 0.749 |

*WT = Whole Tumor · TC = Tumor Core · ET = Enhancing Tumor*  
*Final model evaluated on a 20% held-out split. Paper target is from the official BraTS 2017 validation split.*

### HD95 (lower is better)

| Dataset | HD95 WT | HD95 TC | HD95 ET |
|---|---|---|---|
| BraTS 2017 | 9.08 px | 6.81 px | 4.86 px |
| BraTS 2018 | 8.28 px | 6.69 px | 5.99 px |
| BraTS 2019 | 8.72 px | 9.39 px | 9.58 px |

---

## Enhancements

The following changes were made from the baseline pipeline to the final model:

| Enhancement | Baseline | Final Model |
|---|---|---|
| Training duration | 2 epochs | 40 epochs |
| Steps / epoch | 250 train / 60 val | 1,000 train / 250 val |
| Batch size | 2 | 8 |
| Learning rate | 0.086 (unstable) | 0.003 (stable convergence) |
| Preprocessing | On-the-fly NIfTI loading | Offline `.npy` + patient-level cache |
| Crop strategy | Fixed 128×128 center crop | Tumor-aware crop (85% tumor-centered) |
| ET class weight (WCE) | 1.5 | 2.0 |
| BG class weight (WCE) | 0.1 | 0.2 |
| Checkpoint monitor | `val_dice_wt` (max) | `val_loss` (min) |
| GDL formulation | Fixed 1e-6 smoothing | Batch-adaptive frequency weighting |
| Evaluation metrics | Soft Dice only | Soft Dice + Hard Dice + HD95 per region |

---

## Reports

- **Phase 3 Report** (`reports/phase3_report.pdf`) — full methodology, architecture details, and pipeline setup
- **Enhancement Report** (`reports/enhancement_report.docx`) — before/after comparison, final results, and discussion

---

## Reference

Zhang, J., Jiang, Z., Dong, J., Hou, Y., & Liu, B. (2020). *Attention Gate ResU-Net for Automatic MRI Brain Tumor Segmentation.* IEEE Access, 8, 58533–58545.
