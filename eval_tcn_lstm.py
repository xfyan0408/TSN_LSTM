# eval_tcn_lstm.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional


# =========================
# Model (必须与训练一致)
# =========================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.0, norm="layer", activation="relu"):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("activation must be relu/gelu")

        self.norm1 = nn.LayerNorm(out_ch) if norm == "layer" else None
        self.norm2 = nn.LayerNorm(out_ch) if norm == "layer" else None
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _ln(self, y, ln):
        if ln is None:
            return y
        y = y.transpose(1, 2)  # (B,T,C)
        y = ln(y)
        y = y.transpose(1, 2)
        return y

    def forward(self, x):
        y = self.conv1(x)
        y = self._ln(y, self.norm1)
        y = self.act(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self._ln(y, self.norm2)
        y = self.act(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class TCNBackbone(nn.Module):
    def __init__(self, channels: List[int], kernel_size=3, dilations=None, dropout=0.0, norm="layer", activation="relu"):
        super().__init__()
        n_blocks = len(channels) - 1
        if dilations is None:
            dilations = [2 ** i for i in range(n_blocks)]
        assert len(dilations) == n_blocks

        blocks = []
        for i in range(n_blocks):
            blocks.append(TemporalBlock(channels[i], channels[i+1], kernel_size, dilations[i],
                                        dropout=dropout, norm=norm, activation=activation))
        self.net = nn.Sequential(*blocks)
        self.dilations = dilations
        self.kernel_size = kernel_size

    def forward(self, x):
        return self.net(x)

    def receptive_field(self) -> int:
        k = self.kernel_size
        return 1 + sum(2 * (k - 1) * d for d in self.dilations)


class TCNLSTMForecaster(nn.Module):
    def __init__(
        self,
        in_dim=3, out_dim=3, horizon=10,
        d_model=32, embed_activation="relu",
        tcn_layers=5, tcn_kernel_size=3, tcn_dilations=None,
        tcn_dropout=0.1, tcn_norm="layer", tcn_activation="relu",
        lstm_hidden=64, lstm_layers=1, lstm_dropout=0.0,
        mlp_hidden=64, separate_heads=False,
        residual_to_last=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.residual_to_last = residual_to_last
        self.separate_heads = separate_heads

        self.embed = nn.Linear(in_dim, d_model)
        if embed_activation == "relu":
            self.embed_act = nn.ReLU()
        elif embed_activation == "gelu":
            self.embed_act = nn.GELU()
        elif embed_activation == "none":
            self.embed_act = nn.Identity()
        else:
            raise ValueError("embed_activation must be relu/gelu/none")

        channels = [d_model] + [d_model] * tcn_layers
        self.tcn = TCNBackbone(channels, kernel_size=tcn_kernel_size, dilations=tcn_dilations,
                               dropout=tcn_dropout, norm=tcn_norm, activation=tcn_activation)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        def make_mlp(out_size):
            return nn.Sequential(
                nn.Linear(lstm_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, out_size),
            )

        if separate_heads:
            self.head_b = make_mlp(horizon)
            self.head_c = make_mlp(horizon)
            self.head_m = make_mlp(horizon)
        else:
            self.head = make_mlp(horizon * out_dim)

    def forward(self, x):  # (B,T,3)
        B, T, D = x.shape
        assert D == self.in_dim

        z = self.embed_act(self.embed(x))     # (B,T,d_model)
        z = z.transpose(1, 2)                 # (B,d_model,T)
        z = self.tcn(z)                       # (B,d_model,T)
        z = z.transpose(1, 2)                 # (B,T,d_model)

        _, (h, _) = self.lstm(z)
        h_last = h[-1]                        # (B,lstm_hidden)

        if self.separate_heads:
            yb = self.head_b(h_last).unsqueeze(-1)
            yc = self.head_c(h_last).unsqueeze(-1)
            ym = self.head_m(h_last).unsqueeze(-1)
            y = torch.cat([yb, yc, ym], dim=-1)  # (B,H,3)
        else:
            y = self.head(h_last).view(B, self.horizon, self.out_dim)

        if self.residual_to_last:
            base = x[:, -1:, :].repeat(1, self.horizon, 1)
            y = base + y

        return y


# =========================
# Helpers
# =========================
def make_windows(arr, T, H):
    N = arr.shape[0]
    X, Y = [], []
    for i in range(T, N - H + 1):
        X.append(arr[i - T:i])
        Y.append(arr[i:i + H])
    return np.stack(X), np.stack(Y)

def infer_dt_seconds(ts: pd.Series) -> float:
    d = ts.diff().dropna().dt.total_seconds().to_numpy()
    if len(d) == 0:
        return 0.1
    dt = float(np.median(d))
    if dt <= 0 or dt > 10:
        dt = 0.1
    return dt

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
    return np.concatenate(out, axis=0)  # (N,H,3)

def metrics(pred, true):
    err = pred - true
    mse_all = float((err ** 2).mean())
    mae_all = float(np.abs(err).mean())
    mse_dim = (err ** 2).mean(axis=(0, 1))     # (3,)
    mae_dim = np.abs(err).mean(axis=(0, 1))    # (3,)
    mse_h   = (err ** 2).mean(axis=(0, 2))     # (H,)
    mae_h   = np.abs(err).mean(axis=(0, 2))    # (H,)
    return mse_all, mae_all, mse_dim, mae_dim, mse_h, mae_h


# =========================
# Main eval
# =========================
def main():
    CSV = "data_100ms.csv"
    CKPT = "best_tcn_lstm.pt"
    PLOT_DIR = "plots_tcn_lstm"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load ckpt
    try:
        ckpt = torch.load(CKPT, map_location=device)
    except Exception:
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)

    model_kwargs = ckpt["model_kwargs"]
    T = int(ckpt["T"])
    H = int(ckpt["H"])
    cols = ckpt.get("cols", ["Bandwidth", "CPU", "MEM"])

    mu = ckpt["mu"].detach().cpu().numpy().astype(np.float32)
    std = ckpt["std"].detach().cpu().numpy().astype(np.float32)

    model = TCNLSTMForecaster(**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # load data
    df = pd.read_csv(CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    dt = infer_dt_seconds(df["ts"])

    # 兼容列名：如果 ckpt cols 不在 CSV，就 fallback 自动探测
    if all(c in df.columns for c in cols):
        use_cols = cols
    else:
        if "Bandwidth" in df.columns:
            use_cols = ["Bandwidth", "CPU", "MEM"]
        elif "B" in df.columns:
            use_cols = ["B", "CPU", "MEM"]
        else:
            raise ValueError("CSV里找不到 Bandwidth 或 B 列")
    series = df[use_cols].to_numpy(np.float32)

    # split (与训练一致：70/15/15)
    N = len(series)
    i_train = int(N * 0.70)
    i_val   = int(N * 0.85)

    train_raw = series[:i_train]
    val_raw   = series[i_train:i_val]
    test_raw  = series[i_val:]

    # normalize with ckpt mu/std (与训练一致)
    train = (train_raw - mu) / (std + 1e-6)
    val   = (val_raw   - mu) / (std + 1e-6)
    test  = (test_raw  - mu) / (std + 1e-6)

    # windows on test (with context)
    val_ctx  = np.concatenate([train[-T:], val], axis=0)
    test_ctx = np.concatenate([val_ctx[-T:], test], axis=0)
    X_test, Y_test = make_windows(test_ctx, T, H)

    # predict
    Yp = predict_all(model, X_test, device)

    # baseline: repeat last value
    Yb = np.repeat(X_test[:, -1:, :], repeats=H, axis=1)

    # metrics normalized
    m_mse, m_mae, m_mse_dim, m_mae_dim, m_mse_h, m_mae_h = metrics(Yp, Y_test)
    b_mse, b_mae, b_mse_dim, b_mae_dim, b_mse_h, b_mae_h = metrics(Yb, Y_test)

    print("=== Normalized metrics on TEST ===")
    print(f"[MODEL]    MSE={m_mse:.6f}  MAE={m_mae:.6f}")
    print(f"[BASELINE] MSE={b_mse:.6f}  MAE={b_mae:.6f}")
    print("Dim order:", use_cols)
    print("[MODEL]    MSE_dim:", m_mse_dim, "MAE_dim:", m_mae_dim)
    print("[BASELINE] MSE_dim:", b_mse_dim, "MAE_dim:", b_mae_dim)
    print("MSE by horizon (h=1..H):")
    print("MODEL   :", m_mse_h)
    print("BASELINE:", b_mse_h)

    # metrics raw-scale (de-normalize)
    Yp_raw = Yp * std + mu
    Yt_raw = Y_test * std + mu
    Yb_raw = Yb * std + mu

    rm_mse, rm_mae, rm_mse_dim, rm_mae_dim, rm_mse_h, rm_mae_h = metrics(Yp_raw, Yt_raw)
    rb_mse, rb_mae, rb_mse_dim, rb_mae_dim, rb_mse_h, rb_mae_h = metrics(Yb_raw, Yt_raw)

    print("\n=== Raw-scale metrics on TEST (original units) ===")
    print(f"[MODEL]    MSE={rm_mse:.6f}  MAE={rm_mae:.6f}")
    print(f"[BASELINE] MSE={rb_mse:.6f}  MAE={rb_mae:.6f}")
    print("Dim order:", use_cols)
    print("[MODEL]    MSE_dim:", rm_mse_dim, "MAE_dim:", rm_mae_dim)
    print("[BASELINE] MSE_dim:", rb_mse_dim, "MAE_dim:", rb_mae_dim)

    # plots (optional)
    try:
        import matplotlib.pyplot as plt
        os.makedirs(PLOT_DIR, exist_ok=True)

        # 画几个样本：history + true + pred
        sample_ids = [0, len(X_test)//3, 2*len(X_test)//3]
        times_hist = np.arange(-T, 0) * dt
        times_fut  = np.arange(1, H+1) * dt

        for sid in sample_ids:
            x_hist = X_test[sid] * std + mu
            y_true = Y_test[sid] * std + mu
            y_pred = Yp[sid] * std + mu

            for d, name in enumerate(use_cols):
                plt.figure()
                plt.plot(times_hist, x_hist[:, d], label="history")
                plt.plot(times_fut,  y_true[:, d], label="true")
                plt.plot(times_fut,  y_pred[:, d], label="pred")
                plt.title(f"TCN+LSTM | sample {sid} | {name}")
                plt.xlabel("relative time (s)")
                plt.ylabel(name)
                plt.grid(alpha=0.3, linestyle="--")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f"sample{sid}_{name}.png"), dpi=150)
                plt.close()

        # 画 horizon 误差曲线（raw-scale）
        plt.figure()
        plt.plot(np.arange(1, H+1), rm_mse_h, label="model")
        plt.plot(np.arange(1, H+1), rb_mse_h, label="baseline")
        plt.title("Raw-scale MSE vs horizon (TCN+LSTM)")
        plt.xlabel("horizon step (1..H)")
        plt.ylabel("MSE")
        plt.xticks(np.arange(1, H + 1))
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "mse_by_horizon.png"), dpi=150)
        plt.close()

        print(f"\nSaved plots to ./{PLOT_DIR}/")
    except ImportError:
        print("\nmatplotlib not installed, skip plots. (pip install matplotlib)")

if __name__ == "__main__":
    main()
