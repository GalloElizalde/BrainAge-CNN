import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import MRIDataset
from model import BrainAgeCNN


# Config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#ckpt_path = "brainage_cnn.pt"   # path to checkpoint

# Load model
#model = BrainAgeCNN()
#model.load_state_dict(torch.load(ckpt_path))

# Define test subject ids 
n_subjects = 351      # Current total subjects = 351
ids = np.arange(1, n_subjects + 1)

# Dataset and Loader
data = MRIDataset(ids)

batch_size = 10
data_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"))

# Get ages from data
ages = []
for _, y in data_loader:
    ages.append(y.numpy())

ages = np.concatenate(ages)

# Get min, max, mean and total number
total_ages = ages.shape
min_age = np.min(ages)
max_age = np.max(ages)
mean_age = np.mean(ages)

print(f"Mean age = {mean_age:.2f} |", f"Max age = {max_age} |", f"Min age = {min_age} |")

# Create histogram with rounded ages
ages_binned = (ages // 10) * 10
mean_age_binned = np.mean(ages)
bins = np.arange(ages_binned.min() - 5, ages_binned.max() + 15,10)  # centered bins

plt.title("Distribution of Subject Ages")
plt.hist(ages_binned, bins = bins, label = "ages")
plt.axvline(mean_age_binned, linestyle = "--", color = "k", label = f"mean = {mean_age_binned:.2f}")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y")
plt.savefig("distribution_ages.png", dpi = 150)
plt.show()