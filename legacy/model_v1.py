# train_forecaster.py
# pip install torch pandas numpy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    def __init__(self, arr, T, H):
        self.arr = arr.astype(np.float32)
        self.T, self.H = T, H
        self.n = len(arr) - (T + H) + 1
        if self.n <= 0:
            raise ValueError("Not enough rows for T+H")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.arr[i:i + self.T]
        y = self.arr[i + self.T:i + self.T + self.H]
        return torch.from_numpy(x), torch.from_numpy(y)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation):
        super().__init__()
        self.k, self.d = k, dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=0)

    def forward(self, x):
        left = (self.k - 1) * self.d
        x = F.pad(x, (left, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, k, d, dropout):
        super().__init__()
        self.cc = CausalConv1d(ch, ch, k, d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.drop(self.act(self.cc(x)))
        return x + y


class TCN(nn.Module):
    def __init__(self, ch, k, L, dropout):
        super().__init__()
        self.net = nn.Sequential(*[TCNBlock(ch, k, 2 ** i, dropout) for i in range(L)])

    def forward(self, x):
        return self.net(x)


class Forecaster(nn.Module):
    def __init__(self, H, d_model=32, tcn_layers=6, k=3, lstm_hidden=32, head_hidden=64, dropout=0.1):
        super().__init__()
        self.H = H
        self.mix = nn.Conv1d(3, d_model, kernel_size=1)
        self.mix_act = nn.ReLU()
        self.tcn = TCN(d_model, k, tcn_layers, dropout)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, batch_first=True)

        def head():
            return nn.Sequential(
                nn.Linear(lstm_hidden, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, H),
            )

        self.h_bw, self.h_cpu, self.h_mem = head(), head(), head()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mix_act(self.mix(x))
        x = self.tcn(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = h[-1]

        bw = self.h_bw(h)
        cpu = self.h_cpu(h)
        mem = self.h_mem(h)
        y = torch.stack([bw, cpu, mem], dim=-1)
        return y


def train_epoch(model, loader, opt, device):
    model.train()
    s, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        s += loss.item() * x.size(0)
        n += x.size(0)
    return s / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    s, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        s += loss.item() * x.size(0)
        n += x.size(0)
    return s / max(n, 1)


if __name__ == "__main__":
    df = pd.read_csv("metrics.csv")
    cols = ["bw", "cpu", "mem"]
    raw = df[cols].to_numpy(dtype=np.float32)

    split = int(len(raw) * 0.8)
    tr_raw, va_raw = raw[:split], raw[split:]

    mean = tr_raw.mean(axis=0)
    std = tr_raw.std(axis=0) + 1e-8
    tr = (tr_raw - mean) / std
    va = (va_raw - mean) / std

    T, H = 64, 16
    tr_ds = WindowDataset(tr, T, H)
    va_ds = WindowDataset(va, T, H)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Forecaster(H=H, d_model=32, tcn_layers=6, k=3, lstm_hidden=32, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = 1e18
    for ep in range(30):
        tr_loss = train_epoch(model, tr_ld, opt, device)
        va_loss = eval_epoch(model, va_ld, device)
        print(f"ep={ep:03d} train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best:
            best = va_loss
            torch.save(
                {"state": model.state_dict(), "mean": mean, "std": std, "T": T, "H": H},
                "forecaster.pt",
            )

    print("saved best to forecaster.pt")
