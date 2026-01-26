import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MRIDataset
from model.model import BrainAgeCNN
from data.data_loader import load_mri_slices_and_age  


##  FOR COMPARISON ONLY================
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#=============================================

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define subject ids to load dataset
n_subjects = 351  # Current total subjects = 351
ids = np.arange(1, n_subjects + 1)

# Randomize ids
rng = np.random.default_rng(seed)
rng.shuffle(ids)

# Split ids into train(70%) and validation(15%)
train_frac = 0.70
n_max_idx = int(train_frac * len(ids))

train_ids = ids[:n_max_idx]
val_ids = ids[n_max_idx: int(len(ids) - (len(ids) * 0.15))]

# Train comparison:
slices = [1,3,5,9,15]
for n_slices in slices:

    # Load dataset
    data_train = MRIDataset(train_ids, n_slices)
    data_val = MRIDataset(val_ids, n_slices)

    # Loaders
    batch_size = 6
    use_cuda = (device.type == "cuda")

    loader_train = DataLoader(data_train, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=use_cuda)
    loader_val = DataLoader(data_val, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=use_cuda)

    # Model + hyperparameters
    model = BrainAgeCNN(in_channels=n_slices).to(device)   # move model to GPU
    n_epochs = 150
    learning_rate = 3e-4

    # Loss function + optimizer  (3 for comparison)
    loss_function_mae = nn.L1Loss()
    loss_function_mse = nn.MSELoss()
    loss_function_huber = nn.SmoothL1Loss(beta=5.0)  # Huber

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # ======================================= TRAINING =================================
    print(f"Training in progress for {n_slices} slices and {n_epochs} epochs...")
    MSE_history_train = []
    MAE_history_train = []
    huber_history_train = []
    best_val_mae = float("inf")  # to save best model later

    for epoch in range(n_epochs):

        # Train 
        model.train()
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_huber = 0.0

        for X, y in loader_train:
            # move batch to device
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1).float()  # age to float 

            pred = model(X).view(-1)

            mse = loss_function_mse(pred, y)
            mae = loss_function_mae(pred, y)
            huber = loss_function_huber(pred, y)

            loss = huber

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse += mse.item() * X.size(0)
            epoch_mae  += mae.item() * X.size(0)
            epoch_huber  += huber.item() * X.size(0)

        epoch_mse /= len(loader_train.dataset)
        epoch_mae  /= len(loader_train.dataset)
        epoch_huber  /= len(loader_train.dataset)

        MSE_history_train.append(epoch_mse)
        MAE_history_train.append(epoch_mae)
        huber_history_train.append(epoch_huber)

        #==================================== VALIDATION ==================================
        model.eval()
        val_huber = 0.0
        val_mae = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for X, y in loader_val:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(-1).float()

                pred = model(X).view(-1)

                huber = loss_function_huber(pred, y)
                mae = loss_function_mae(pred, y)

                val_huber += huber.item() * X.size(0)
                val_mae  += mae.item() * X.size(0)

        val_huber /= len(loader_val.dataset)
        val_mae  /= len(loader_val.dataset)

        # check vae to save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"../model/best_s{n_slices}.pt")


        # ========================== LOG ========================
        print(
        f"Epoch {epoch+1:03d} | "
        f"Train: MAE={epoch_mae:6.2f}, Huber={epoch_huber:5.2f} | "
        f"Val:   MAE={val_mae:6.2f}, Huber={val_huber:5.2f}")


    # Make Plot of loss function 
    
    plt.plot(huber_history_train, label=f"{n_slices} MRI slices")
plt.title("Huber Loss Function Evolucion (train)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.savefig("./loss_evolution_huber_train.png", dpi = 150)
plt.show()

