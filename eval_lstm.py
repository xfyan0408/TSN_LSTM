import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- model ----------
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

    def forward(self, x):  # (B,T,3)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        y = self.head(h_last)
        return y.view(-1, self.H, self.out_dim)

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
    mse_dim = (err ** 2).mean(axis=(0, 1))  # (3,)
    mae_dim = np.abs(err).mean(axis=(0, 1)) # (3,)
    mse_h   = (err ** 2).mean(axis=(0, 2))  # (H,)
    mae_h   = np.abs(err).mean(axis=(0, 2)) # (H,)
    return mse_all, mae_all, mse_dim, mae_dim, mse_h, mae_h

def main():
    CSV = "data_100ms.csv"
    CKPT = "best_lstm.pt"
    # CKPT = "best_tcn_lstm.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load ckpt -----
    try:
        ckpt = torch.load(CKPT, map_location=device)
    except Exception:
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)

    T = int(ckpt.get("T", 40))
    H = int(ckpt.get("H", 10))
    mu = ckpt["mu"].detach().cpu().numpy().astype(np.float32)   # (3,)
    std = ckpt["std"].detach().cpu().numpy().astype(np.float32) # (3,)

    # 训练时 hidden/num_layers 你是 64/1（如果你改过，这里也要改）
    hidden = 64
    num_layers = 1

    model = LSTMForecaster(in_dim=3, hidden=hidden, num_layers=num_layers, H=H, out_dim=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ----- load data -----
    df = pd.read_csv(CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    dt = infer_dt_seconds(df["ts"])

    # 兼容列名 Bandwidth 或 B
    if "Bandwidth" in df.columns:
        cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列")

    series = df[cols].to_numpy(np.float32)
    N = len(series)
    i_train = int(N * 0.70)
    i_val   = int(N * 0.85)

    train_raw = series[:i_train]
    val_raw   = series[i_train:i_val]
    test_raw  = series[i_val:]

    # 用 ckpt 里的 mu/std（确保与训练一致）
    train = (train_raw - mu) / (std + 1e-6)
    val   = (val_raw   - mu) / (std + 1e-6)
    test  = (test_raw  - mu) / (std + 1e-6)

    # windows (test)
    val_ctx  = np.concatenate([train[-T:], val], axis=0)
    test_ctx = np.concatenate([val_ctx[-T:], test], axis=0)
    X_test, Y_test = make_windows(test_ctx, T, H)

    # ----- model pred -----
    Yp = predict_all(model, X_test, device)

    # ----- baseline pred (persistence) -----
    Yb = np.repeat(X_test[:, -1:, :], repeats=H, axis=1)

    # ----- eval in normalized space -----
    m_mse, m_mae, m_mse_dim, m_mae_dim, m_mse_h, m_mae_h = metrics(Yp, Y_test)
    b_mse, b_mae, b_mse_dim, b_mae_dim, b_mse_h, b_mae_h = metrics(Yb, Y_test)

    print("=== Normalized metrics on TEST ===")
    print(f"[MODEL]   MSE={m_mse:.6f}  MAE={m_mae:.6f}")
    print(f"[BASELINE]MSE={b_mse:.6f}  MAE={b_mae:.6f}")
    print("Dim order:", cols)
    print("[MODEL]   MSE_dim:", m_mse_dim, "MAE_dim:", m_mae_dim)
    print("[BASELINE]MSE_dim:", b_mse_dim, "MAE_dim:", b_mae_dim)
    print("MSE by horizon (h=1..H):")
    print("MODEL   :", m_mse_h)
    print("BASELINE:", b_mse_h)

    # ----- eval in raw space (反标准化) -----
    Yp_raw = Yp * std + mu
    Yt_raw = Y_test * std + mu
    Yb_raw = Yb * std + mu

    rm_mse, rm_mae, rm_mse_dim, rm_mae_dim, rm_mse_h, rm_mae_h = metrics(Yp_raw, Yt_raw)
    rb_mse, rb_mae, rb_mse_dim, rb_mae_dim, rb_mse_h, rb_mae_h = metrics(Yb_raw, Yt_raw)

    print("\n=== Raw-scale metrics on TEST (original units) ===")
    print(f"[MODEL]   MSE={rm_mse:.6f}  MAE={rm_mae:.6f}")
    print(f"[BASELINE]MSE={rb_mse:.6f}  MAE={rb_mae:.6f}")
    print("Dim order:", cols)
    print("[MODEL]   MSE_dim:", rm_mse_dim, "MAE_dim:", rm_mae_dim)
    print("[BASELINE]MSE_dim:", rb_mse_dim, "MAE_dim:", rb_mae_dim)

    # ----- optional plots -----
    try:
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)

        # 画 3 个样本（你可以改索引）
        sample_ids = [0, len(X_test)//3, 2*len(X_test)//3]
        for sid in sample_ids:
            x_hist = (X_test[sid] * std + mu)  # (T,3)
            y_true = (Y_test[sid] * std + mu)  # (H,3)
            y_pred = (Yp[sid] * std + mu)      # (H,3)

            t0 = df["ts"].iloc[i_val]  # test大概起点，用作参考不必精确
            times_hist = np.arange(-T, 0) * dt
            times_fut  = np.arange(1, H+1) * dt

            # 每个维度单独一张图（避免复杂）
            for d, name in enumerate(cols):
                plt.figure()
                plt.plot(times_hist, x_hist[:, d], label="history")
                plt.plot(times_fut, y_true[:, d], label="true")
                plt.plot(times_fut, y_pred[:, d], label="pred")
                plt.title(f"sample {sid} | {name}")
                plt.xlabel("time (s) relative")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"plots/sample{sid}_{name}.png", dpi=150)
                plt.close()

        # 画 horizon 误差曲线
        plt.figure()
        plt.plot(np.arange(1, H+1), rm_mse_h, label="model")
        plt.plot(np.arange(1, H+1), rb_mse_h, label="baseline")
        plt.title("MSE vs horizon (raw scale)")
        plt.xlabel("horizon step (1..H)")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/mse_by_horizon.png", dpi=150)
        plt.close()

        print("\nSaved plots to ./plots/")
    except ImportError:
        print("\nmatplotlib not installed, skip plots. (pip install matplotlib)")

if __name__ == "__main__":
    main()
