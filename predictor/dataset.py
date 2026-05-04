import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_PATH, PRED_HORIZON, RESOURCE_COLUMNS, VAL_RATIO, WINDOW_SIZE, TRAIN_RATIO


class ResourceDataset(Dataset):
    def __init__(self, values, window_size, pred_horizon, mean, std):
        self.values = ((values - mean) / std).astype(np.float32)
        self.window_size = window_size
        self.pred_horizon = pred_horizon

    def __len__(self):
        return len(self.values) - self.window_size - self.pred_horizon + 1

    def __getitem__(self, idx):
        x = self.values[idx:idx + self.window_size]
        y = self.values[idx + self.window_size:idx + self.window_size + self.pred_horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_resource_values(csv_path=DATA_PATH, columns=RESOURCE_COLUMNS):
    df = pd.read_csv(csv_path)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    if all(c in df.columns for c in columns):
        use_cols = list(columns)
    elif all(c in df.columns for c in ("bandwidth", "cpu", "memory")):
        use_cols = ["bandwidth", "cpu", "memory"]
    else:
        raise ValueError("CSV must contain Bandwidth/CPU/MEM or bandwidth/cpu/memory columns")
    return df, df[use_cols].to_numpy(np.float32), use_cols


def build_datasets(
    csv_path=DATA_PATH,
    window_size=WINDOW_SIZE,
    pred_horizon=PRED_HORIZON,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
):
    df, values, columns = load_resource_values(csv_path)
    n = len(values)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train_raw = values[:i_train]
    val_raw = values[i_train:i_val]
    test_raw = values[i_val:]

    mean = train_raw.mean(axis=0).astype(np.float32)
    std = train_raw.std(axis=0).astype(np.float32) + 1e-6

    val_ctx = np.concatenate([train_raw[-window_size:], val_raw], axis=0)
    test_ctx = np.concatenate([val_ctx[-window_size:], test_raw], axis=0)

    train_set = ResourceDataset(train_raw, window_size, pred_horizon, mean, std)
    val_set = ResourceDataset(val_ctx, window_size, pred_horizon, mean, std)
    test_set = ResourceDataset(test_ctx, window_size, pred_horizon, mean, std)

    return train_set, val_set, test_set, mean, std, columns, df
