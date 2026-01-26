import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data.dataset import MRIDataset
from model.model import BrainAgeCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# match training split
seed = 42
rng = np.random.default_rng(seed)
ids = np.arange(1, 351 + 1)
#rng.shuffle(ids) 

n = len(ids)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
test_ids = ids[n_train + n_val:]

# bins per age
edges = [0, 30, 45, 60, 75, 120]

# Evaluate 5 models
slices = [1, 3, 5, 9, 15]
for n_slices in slices:

    ckpt_path = f"./model/best_s{n_slices}.pt"

    model = BrainAgeCNN(in_channels=n_slices).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_data = MRIDataset(test_ids, n_slices)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    y_pred, y_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1).float()
            pred = model(X).view(-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    error = y_pred - y_true
    abs_error = np.abs(error)

    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    bias = error.mean()
    r = np.corrcoef(y_true, y_pred)[0, 1]
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n[TEST s={n_slices}] MAE {mae:.2f} | RMSE {rmse:.2f} | Bias {bias:.2f} | r {r:.2f} | R2 {r2:.2f}")

    # MAE per binned age
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_true > lo) & (y_true <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            print(f"  bin [{lo},{hi}] n=0")
            continue
        mae_bin = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        print(f"  bin [{lo},{hi}] n={n_bin:3d} MAE={mae_bin:.2f}")

    # 10 worst errors
    worst_idx = np.argsort(abs_error)[::-1][:10]
    for i in worst_idx:
        print(
            f"  worst | idx={i:3d} | true={y_true[i]:5.1f} | pred={y_pred[i]:5.1f} | "
            f"|err|={abs_error[i]:6.1f}"
        )

    # histogram of errors
    plt.figure()
    plt.title(f"Test Error (pred - true) | s={n_slices}")
    plt.hist(error, bins="auto")
    plt.xlabel("Error (years)")
    plt.ylabel("Count")
    plt.grid(axis="y")
    plt.savefig(f"./eval/test_error_hist_s{n_slices}.png", dpi=150)
    plt.close()

    # scatter plot
    plt.figure()
    plt.title(f"Test: Predicted vs True Age | s={n_slices}")
    plt.scatter(y_true, y_pred, s=10)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "--k", label="ideal")
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.grid()
    plt.legend()
    plt.savefig(f"./eval/test_scatter_s{n_slices}.png", dpi=150)
    plt.close()