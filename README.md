# BrainAge CNN

This project predicts **chronological age from brain MRI** using a convolutional neural network (CNN).

The goal is to build a **simple and reproducible baseline** for brain age regression using a small number of MRI slices. This is **not a state-of-the-art model**, but a clean end-to-end medical imaging + ML pipeline.

## What the project does

- Loads structural MRI data from the OASIS dataset
- Extracts 15 axial slices from the central part of each brain volume
- Stacks slices as channels and feeds them to a 2D CNN
- Trains the model to predict age in years

## Dataset

- Dataset: OASIS
- Number of subjects: 351
- Target variable: chronological age

### Slice selection

- 15 axial slices per subject
- Uniformly sampled between 25% and 75% of the Z-axis
- Input shape: (15, H, W)

## Preprocessing

- MRI volumes loaded with nibabel
- Per-subject z-score normalization
- No skull stripping or heavy preprocessing

## Model

- 2D CNN implemented in PyTorch
- 3 convolutional blocks (Conv + BatchNorm + ReLU)
- Global average pooling
- Linear regression head

## Training

- Train / validation split: 70% / 15%
- Optimizer: AdamW
- Loss used for training: Huber loss
- Metrics monitored: MAE and MSE
- Fixed random seeds for reproducibility

## Evaluation

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Bias (mean error)
- Pearson correlation
- R2 score
- Error histograms
- Predicted vs true age scatter plot
- MAE by age bins

## Repository structure

BrainAge-CNN/
├── data_loader.py
├── dataset.py
├── model.py
├── train.py
├── eval.py
└── README.md

## Notes

This project is intended as a baseline and learning project for medical imaging and machine learning. It is designed to be easy to understand, modify, and extend.
