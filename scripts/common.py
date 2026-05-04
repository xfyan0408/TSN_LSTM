import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_windows(arr, T, H):
    n = arr.shape[0]
    xs, ys = [], []
    for i in range(T, n - H + 1):
        xs.append(arr[i - T:i])
        ys.append(arr[i:i + H])
    return np.stack(xs), np.stack(ys)


def infer_dt_seconds(ts: pd.Series) -> float:
    d = ts.diff().dropna().dt.total_seconds().to_numpy()
    if len(d) == 0:
        return 0.1
    dt = float(np.median(d))
    if dt <= 0 or dt > 10:
        return 0.1
    return dt


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


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


@torch.no_grad()
def predict_all(model, X, device, batch_size=512):
    ds = TensorDataset(torch.from_numpy(X))
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    for (xb,) in ld:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        out.append(pred)
    return np.concatenate(out, axis=0)


def metrics(pred, true):
    err = pred - true
    mse_all = float((err ** 2).mean())
    mae_all = float(np.abs(err).mean())
    mse_dim = (err ** 2).mean(axis=(0, 1))
    mae_dim = np.abs(err).mean(axis=(0, 1))
    mse_h = (err ** 2).mean(axis=(0, 2))
    mae_h = np.abs(err).mean(axis=(0, 2))
    return mse_all, mae_all, mse_dim, mae_dim, mse_h, mae_h
