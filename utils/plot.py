import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import MRIDataset


# Define subject ids to load dataset

n_subjects = 246 # Total subjects = 246
ids = np.arange(1, n_subjects + 1)
np.random.shuffle(ids)   # Shuffle ids 


# Split ids into train(80%) and validation(20%) ids

train_frac = 0.8
n_max_idx = int( train_frac * len(ids) ) 

train_ids = ids[:n_max_idx]
val_ids = ids[n_max_idx:]


# Load dataset

data_train = MRIDataset(train_ids)
data_val = MRIDataset(val_ids)


# Data loader

loader_train = DataLoader(data_train, batch_size = 8, num_workers = 0) 
loader_val = DataLoader(data_val, batch_size = 8, num_workers = 0)


X, y = next(iter(loader_train))

print(X.shape, y)

plt.figure(figsize=(12,12))
for i in range(3):
    plt.subplot(3,3,3*i + 1)
    plt.title(f"Patient {train_ids[i]}, Age = {y[i].item():.0f}, Section 81/272")
    plt.imshow(X[i,0,:,:].T, origin = "lower", cmap = "turbo")

    plt.subplot(3,3,3*i + 2)
    plt.title(f"Patient {train_ids[i]}, Age = {y[i].item():.0f}, Section 136/272")
    plt.imshow(X[i,1,:,:].T, origin = "lower", cmap = "turbo")

    plt.subplot(3,3,3*i + 3)
    plt.title(f"Patient {train_ids[i]}, Age = {y[i].item():.0f}, Section 163/272")
    plt.imshow(X[i,2,:,:].T, origin = "lower", cmap = "turbo")

plt.tight_layout()
plt.savefig("patients_MRI_example.png", dpi = 150)
plt.show()

