# BrainAge CNN

This project predicts **chronological age from brain MRI** using a convolutional neural network (CNN).

The goal is to build a **simple, reproducible baseline** for predicting chronological age from brain MRI, while explicitly studying how the **number of input slices** affects regression performance.

The aim is create an interpretable end-to-end machine learning pipeline for medical imaging, prioritizing transparency and reproducibility over state-of-the-art performance.

## Problem formulation

- Task: supervised regression
- Input: structural T1-weighted brain MRI
- Output: chronological age (continuous, in years)

---

## Pipeline overview

For each subject:

- Load a 3D MRI volume
- Extract a fixed number of axial slices from the central brain region
- Stack slices as channels
- Train a CNN to regress age

The project evaluates multiple configurations with different numbers of slices.

---

## Dataset

- Dataset: OASIS (Open Access Series of Imaging Studies)
- Number of subjects: 351
- Target variable: chronological age

---

## Slice sampling

- Axial slices are sampled uniformly between **25% and 75%** of the Z-axis
- The number of slices is a configurable parameter

Experiments are run with:
- 1 slice
- 3 slices
- 5 slices
- 9 slices
- 15 slices

Input tensor shape:
- `(n_slices, H, W)`

This allows studying the trade-off between spatial information and model complexity.

---

## Preprocessing

- MRI volumes loaded using nibabel
- Per-subject z-score normalization
- No skull stripping or spatial normalization
- Same preprocessing for all slice configurations

---

## Model

- 2D convolutional neural network (PyTorch)
- Three convolutional blocks:
  - Conv2D + BatchNorm + ReLU
- Global average pooling
- Linear regression head
- Same architecture used for all experiments (only input channels change)

---

## Training setup

- Supervised regression
- Loss function: Huber loss
- Optimizer: AdamW
- Train / validation / test split
- Fixed random seeds for reproducibility
- Best model selected based on validation performance

---

## Evaluation

Model performance is evaluated on a held-out test set using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Bias (mean signed error)
- Pearson correlation coefficient
- R2 score
- Error distribution histograms
- Predicted vs true age scatter plots
- MAE stratified by age bins

Results are compared across different numbers of input slices.

---

## Repository structure

BrainAge-CNN/
├── data_loader.py
├── dataset.py
├── model.py
├── train.py
├── eval.py
├── plot.py
├── data_visualization.py
└── README.md

---


## Dataset reference

This project uses data from the **OASIS (Open Access Series of Imaging Studies)** dataset.

Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L.  
*Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle-aged, nondemented, and demented older adults.*  
Journal of Cognitive Neuroscience, 19(9), 1498–1507, 2007.

https://www.oasis-brains.org

