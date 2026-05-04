import numpy as np
import pandas as pd
import torch


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def save_checkpoint(path, model, mean, std, columns, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "columns": columns,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def infer_dt_seconds(ts):
    d = pd.to_datetime(ts).diff().dropna().dt.total_seconds().to_numpy()
    if len(d) == 0:
        return 1.0
    dt = float(np.median(d))
    return dt if dt > 0 else 1.0
