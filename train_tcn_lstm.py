# train_tcn_lstm_allinone.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional


# =========================================================
# 1) Model: ChannelMix(Linear) -> TCN -> LSTM -> Head
# =========================================================

class CausalConv1d(nn.Module):
    """只在左侧 padding 的因果卷积：保证不看未来。"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))  # (left, right)
        return self.conv(x)


class TemporalBlock(nn.Module):
    """TCN 残差块：两层因果卷积 + 残差连接。"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        norm: str = "layer",     # "none" | "layer"
        activation: str = "relu" # "relu" | "gelu"
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.norm1 = nn.LayerNorm(out_ch) if norm == "layer" else None
        self.norm2 = nn.LayerNorm(out_ch) if norm == "layer" else None

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _apply_ln(self, y: torch.Tensor, ln: Optional[nn.LayerNorm]) -> torch.Tensor:
        if ln is None:
            return y
        # y: (B,C,T) -> (B,T,C) -> LN -> (B,C,T)
        y = y.transpose(1, 2)
        y = ln(y)
        y = y.transpose(1, 2)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        y = self.conv1(x)
        y = self._apply_ln(y, self.norm1)
        y = self.act(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self._apply_ln(y, self.norm2)
        y = self.act(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class TCNBackbone(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        n_blocks = len(channels) - 1
        if dilations is None:
            dilations = [2 ** i for i in range(n_blocks)]
        assert len(dilations) == n_blocks

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                TemporalBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.dilations = dilations
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,C,T)

    def receptive_field(self) -> int:
        """
        近似感受野：每个 block 两层卷积，增量 2*(k-1)*d
        R = 1 + Σ 2*(k-1)*d_i
        """
        k = self.kernel_size
        return 1 + sum(2 * (k - 1) * d for d in self.dilations)


class TCNLSTMForecaster(nn.Module):
    """
    ChannelMix(Linear) -> TCN -> LSTM -> Head -> (B,H,3)
    """
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        horizon: int = 10,

        # ChannelMix / embedding
        d_model: int = 32,
        embed_activation: str = "relu",  # "relu"|"gelu"|"none"

        # TCN
        tcn_layers: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: Optional[List[int]] = None,
        tcn_dropout: float = 0.1,
        tcn_norm: str = "layer",         # "none"|"layer"
        tcn_activation: str = "relu",    # "relu"|"gelu"

        # LSTM
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,

        # Head
        mlp_hidden: int = 64,
        separate_heads: bool = False,
        residual_to_last: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.residual_to_last = residual_to_last
        self.separate_heads = separate_heads

        # ChannelMix / embedding
        self.embed = nn.Linear(in_dim, d_model)
        if embed_activation == "relu":
            self.embed_act = nn.ReLU()
        elif embed_activation == "gelu":
            self.embed_act = nn.GELU()
        elif embed_activation == "none":
            self.embed_act = nn.Identity()
        else:
            raise ValueError("embed_activation must be relu/gelu/none")

        # TCN backbone (保持通道 d_model)
        channels = [d_model] + [d_model] * tcn_layers
        self.tcn = TCNBackbone(
            channels=channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
            norm=tcn_norm,
            activation=tcn_activation,
        )

        # LSTM aggregator
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        def make_mlp(out_size: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,3)
        B, T, D = x.shape
        assert D == self.in_dim

        # (B,T,3)->(B,T,d_model)
        z = self.embed_act(self.embed(x))

        # TCN expects (B,C,T)
        z = z.transpose(1, 2)    # (B,d_model,T)
        z = self.tcn(z)          # (B,d_model,T)
        z = z.transpose(1, 2)    # (B,T,d_model)

        # LSTM -> h_last
        _, (h, _) = self.lstm(z)
        h_last = h[-1]           # (B,lstm_hidden)

        if self.separate_heads:
            yb = self.head_b(h_last).unsqueeze(-1)   # (B,H,1)
            yc = self.head_c(h_last).unsqueeze(-1)   # (B,H,1)
            ym = self.head_m(h_last).unsqueeze(-1)   # (B,H,1)
            y = torch.cat([yb, yc, ym], dim=-1)      # (B,H,3)
        else:
            y = self.head(h_last).view(B, self.horizon, self.out_dim)

        # residual: baseline(last value repeat) + delta
        if self.residual_to_last:
            base = x[:, -1:, :].repeat(1, self.horizon, 1)
            y = base + y

        return y


# =========================================================
# 2) Training utilities
# =========================================================

def make_windows(arr, T, H):
    N = arr.shape[0]
    X, Y = [], []
    for i in range(T, N - H + 1):
        X.append(arr[i - T:i])
        Y.append(arr[i:i + H])
    return np.stack(X), np.stack(Y)


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


# =========================================================
# 3) Main train
# =========================================================

def main():
    # ===== Config (data/training) =====
    CSV = "data_100ms.csv"
    T, H = 40, 10
    batch_size = 256
    epochs = 30
    lr = 1e-3
    seed = 42

    # ===== Config (model hyperparams) =====
    model_kwargs = dict(
        in_dim=3,
        out_dim=3,
        horizon=H,

        d_model=32,
        embed_activation="relu",

        tcn_layers=5,
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4, 8, 16],  # 感受野覆盖 T=40
        tcn_dropout=0.1,
        tcn_norm="layer",
        tcn_activation="relu",

        lstm_hidden=64,
        lstm_layers=1,
        lstm_dropout=0.0,

        mlp_hidden=64,
        separate_heads=False,
        residual_to_last=True,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ===== Load CSV =====
    df = pd.read_csv(CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    # 兼容列名：Bandwidth 或 B
    if "Bandwidth" in df.columns:
        cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列")

    series = df[cols].to_numpy(np.float32)
    N = len(series)

    # ===== Split by time =====
    i_train = int(N * 0.70)
    i_val   = int(N * 0.85)

    train_raw = series[:i_train]
    val_raw   = series[i_train:i_val]
    test_raw  = series[i_val:]

    # ===== Normalize (train stats only) =====
    mu = train_raw.mean(axis=0).astype(np.float32)
    std = (train_raw.std(axis=0).astype(np.float32) + 1e-6)

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

    # ===== Baseline (persistence) =====
    yhat = np.repeat(X_val[:, -1:, :], repeats=H, axis=1)
    baseline_mse = float(((yhat - Y_val) ** 2).mean())
    print("baseline val MSE:", baseline_mse)

    # ===== Torch loaders =====
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(Y_test))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Model =====
    model = TCNLSTMForecaster(**model_kwargs).to(device)
    print("TCN receptive field:", model.tcn.receptive_field())

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best = 1e18
    loss_fn = nn.SmoothL1Loss(beta=1.0)   # 或 nn.HuberLoss(delta=1.0)
    # loss_fn = nn.L1Loss()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            # loss = ((pred - yb) ** 2).mean()
            loss = loss_fn(pred, yb)  # SmoothL1Loss
            # loss = loss_fn(pred, yb) 

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
                    "mu": torch.tensor(mu, dtype=torch.float32),
                    "std": torch.tensor(std, dtype=torch.float32),
                    "cols": cols,
                    "T": T,
                    "H": H,
                    "model_kwargs": model_kwargs,  # 推理时自动还原模型
                },
                "best_tcn_lstm.pt"
            )

    # ===== Load best & test =====
    try:
        ckpt = torch.load("best_tcn_lstm.pt", map_location=device)
    except Exception:
        ckpt = torch.load("best_tcn_lstm.pt", map_location=device, weights_only=False)

    best_model = TCNLSTMForecaster(**ckpt["model_kwargs"]).to(device)
    best_model.load_state_dict(ckpt["state_dict"])
    test_mse = eval_mse(best_model, test_ld, device)

    print("best val_mse:", best)
    print("test_mse:", test_mse)
    print("saved:", "best_tcn_lstm.pt")


if __name__ == "__main__":
    main()
