import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data_loader import load_mri_slices_and_age

class MRIDataset(Dataset):
    """
    Each item in the dataset corresponds to a single subject and returns:
        - images: torch.Tensor of shape (15, H, W), three axial MRI slices stacked as channels.
        - age: torch.Tensor scalar age of the subject.
    """

    def __init__(self, subject_ids, n_slices: int = 15):
        self.subject_ids = subject_ids
        self.n_slices = n_slices

    def __len__(self): # Number of subjects in the dataset.
        return len(self.subject_ids)

    
    def __getitem__(self, idx):  #Load and return a single sample.

        # Map internal index to actual subject number (OASIS ID)
        subject_number = self.subject_ids[idx]

        # Load MRI slices and age using user-defined loader
        images, age = load_mri_slices_and_age(subject_number, self.n_slices)

        # Convert to float32 (required by PyTorch and faster than float64)
        images = images.astype(np.float32)

        # Per-sample z-score normalization
        # Ensures zero mean and unit variance for each subject
        images = (images - images.mean()) / (images.std() + 1e-8)

        # Convert numpy arrays to PyTorch tensors
        images = torch.from_numpy(images)              # (n_slices, H, W)
        age = torch.tensor(age, dtype=torch.float32)   # scalar

        return images, age
    