import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def make_windows(arr, T, H):
    N = arr.shape[0]
    X, Y = [], []
    for i in range(T, N - H + 1):
        X.append(arr[i - T:i])
        Y.append(arr[i:i + H])
    return np.stack(X), np.stack(Y)

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=3, hidden=64, num_layers=1, H=10, out_dim=3):
        super().__init__()
        self.H, self.out_dim = H, out_dim
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, H * out_dim)
        )

    def forward(self, x):  # (Bandwidth,T,3)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]  # (Bandwidth,hidden)
        y = self.head(h_last)
        return y.view(-1, self.H, self.out_dim)  # (Bandwidth,H,3)

@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        tot += ((pred - yb) ** 2).mean().item()
        n += 1
    return tot / max(1, n)

def main():
    # ===== Config =====
    T, H = 40, 10
    batch_size = 256
    epochs = 30
    lr = 1e-3

    # ===== Load CSV =====
    df = pd.read_csv("data_100ms.csv", parse_dates=["ts"])
    df = df.sort_values("ts")
    series = df[["Bandwidth", "CPU", "MEM"]].to_numpy(np.float32)
    N = len(series)

    # ===== Split by time =====
    i_train = int(N * 0.70)
    i_val   = int(N * 0.85)

    train_raw = series[:i_train]
    val_raw   = series[i_train:i_val]
    test_raw  = series[i_val:]

    # ===== Normalize (train stats only) =====
    mu = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-6

    train = (train_raw - mu) / std
    val   = (val_raw   - mu) / std
    test  = (test_raw  - mu) / std

    # ===== Windows =====
    X_train, Y_train = make_windows(train, T, H)

    val_ctx = np.concatenate([train[-T:], val], axis=0)
    X_val, Y_val = make_windows(val_ctx, T, H)

    test_ctx = np.concatenate([val_ctx[-T:], test], axis=0)
    X_test, Y_test = make_windows(test_ctx, T, H)

    print("X_train/Y_train:", X_train.shape, Y_train.shape)
    print("X_val/Y_val:", X_val.shape, Y_val.shape)
    print("X_test/Y_test:", X_test.shape, Y_test.shape)

    # ===== Baseline: persistence =====
    yhat = np.repeat(X_val[:, -1:, :], repeats=H, axis=1)
    baseline_mse = ((yhat - Y_val) ** 2).mean()
    print("baseline val MSE:", float(baseline_mse))

    # ===== Torch loaders =====
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(Y_test))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMForecaster(in_dim=3, hidden=64, num_layers=1, H=H, out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best = 1e18
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_mse = eval_mse(model, val_ld, device)
        print(f"epoch {ep:02d} | val_mse={val_mse:.6f}")

        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mu": torch.from_numpy(mu),
                    "std": torch.from_numpy(std),
                    "T": T,
                    "H": H,
                },
                "best_lstm.pt"
            )

    ckpt = torch.load("best_lstm.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    
    mu_t  = ckpt["mu"].to(device)   # shape (3,)
    std_t = ckpt["std"].to(device)  # shape (3,)

    test_mse = eval_mse(model, test_ld, device)
    print("best val_mse:", best)
    print("test_mse:", test_mse)
    print("saved:", "best_lstm.pt")
    

if __name__ == "__main__":
    main()
