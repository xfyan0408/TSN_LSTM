import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_PATH, PRED_HORIZON, RESOURCE_COLUMNS, VAL_RATIO, WINDOW_SIZE, TRAIN_RATIO


class ResourceDataset(Dataset):
    def __init__(self, values, window_size, pred_horizon, mean, std):
        # 用训练集的均值和方差做归一化
        self.values = ((values - mean) / std).astype(np.float32)
        self.window_size = window_size
        self.pred_horizon = pred_horizon

    def __len__(self):
        # 可构造的滑动窗口数量
        return len(self.values) - self.window_size - self.pred_horizon + 1

    def __getitem__(self, idx):
        # x 是历史窗口，y 是未来预测目标
        x = self.values[idx:idx + self.window_size]
        y = self.values[idx + self.window_size:idx + self.window_size + self.pred_horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_resource_values(csv_path=DATA_PATH, columns=RESOURCE_COLUMNS):
    # 读取原始 CSV，并按时间排序
    df = pd.read_csv(csv_path)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    # 只使用配置中指定的三列资源指标
    if not all(c in df.columns for c in columns):
        raise ValueError("CSV must contain Bandwidth, CPU, MEM columns")
    use_cols = list(columns)
    return df, df[use_cols].to_numpy(np.float32), use_cols


def build_datasets(
    csv_path=DATA_PATH,
    window_size=WINDOW_SIZE,
    pred_horizon=PRED_HORIZON,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
):
    df, values, columns = load_resource_values(csv_path)

    # 按时间顺序切分，避免未来数据泄漏到训练集
    n = len(values)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train_raw = values[:i_train]
    val_raw = values[i_train:i_val]
    test_raw = values[i_val:]

    mean = train_raw.mean(axis=0).astype(np.float32)
    std = train_raw.std(axis=0).astype(np.float32) + 1e-6

    # 验证/测试集前面补一段历史窗口，保证能构造第一个样本
    val_ctx = np.concatenate([train_raw[-window_size:], val_raw], axis=0)
    test_ctx = np.concatenate([val_ctx[-window_size:], test_raw], axis=0)

    train_set = ResourceDataset(train_raw, window_size, pred_horizon, mean, std)
    val_set = ResourceDataset(val_ctx, window_size, pred_horizon, mean, std)
    test_set = ResourceDataset(test_ctx, window_size, pred_horizon, mean, std)

    return train_set, val_set, test_set, mean, std, columns, df
