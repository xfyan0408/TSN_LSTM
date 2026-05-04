import os

import numpy as np
import pandas as pd
import torch

from models import TCNLSTMForecaster
from scripts.common import infer_dt_seconds, load_checkpoint, make_windows, metrics, predict_all


def main():
    CSV = "data_100ms.csv"
    CKPT = "best_tcn_lstm.pt"
    PLOT_DIR = "plots_tcn_lstm"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = load_checkpoint(CKPT, device)

    model_kwargs = ckpt["model_kwargs"]
    T = int(ckpt["T"])
    H = int(ckpt["H"])
    cols = ckpt.get("cols", ["Bandwidth", "CPU", "MEM"])

    mu = ckpt["mu"].detach().cpu().numpy().astype(np.float32)
    std = ckpt["std"].detach().cpu().numpy().astype(np.float32)

    model = TCNLSTMForecaster(**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = pd.read_csv(CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    dt = infer_dt_seconds(df["ts"])

    if all(c in df.columns for c in cols):
        use_cols = cols
    elif "Bandwidth" in df.columns:
        use_cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        use_cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列")

    series = df[use_cols].to_numpy(np.float32)
    n = len(series)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)

    train_raw = series[:i_train]
    val_raw = series[i_train:i_val]
    test_raw = series[i_val:]

    train = (train_raw - mu) / (std + 1e-6)
    val = (val_raw - mu) / (std + 1e-6)
    test = (test_raw - mu) / (std + 1e-6)

    val_ctx = np.concatenate([train[-T:], val], axis=0)
    test_ctx = np.concatenate([val_ctx[-T:], test], axis=0)
    X_test, Y_test = make_windows(test_ctx, T, H)

    Yp = predict_all(model, X_test, device)
    Yb = np.repeat(X_test[:, -1:, :], repeats=H, axis=1)

    m_mse, m_mae, m_mse_dim, m_mae_dim, m_mse_h, _ = metrics(Yp, Y_test)
    b_mse, b_mae, b_mse_dim, b_mae_dim, b_mse_h, _ = metrics(Yb, Y_test)

    print("=== Normalized metrics on TEST ===")
    print(f"[MODEL]    MSE={m_mse:.6f}  MAE={m_mae:.6f}")
    print(f"[BASELINE] MSE={b_mse:.6f}  MAE={b_mae:.6f}")
    print("Dim order:", use_cols)
    print("[MODEL]    MSE_dim:", m_mse_dim, "MAE_dim:", m_mae_dim)
    print("[BASELINE] MSE_dim:", b_mse_dim, "MAE_dim:", b_mae_dim)
    print("MSE by horizon (h=1..H):")
    print("MODEL   :", m_mse_h)
    print("BASELINE:", b_mse_h)

    Yp_raw = Yp * std + mu
    Yt_raw = Y_test * std + mu
    Yb_raw = Yb * std + mu

    rm_mse, rm_mae, rm_mse_dim, rm_mae_dim, rm_mse_h, _ = metrics(Yp_raw, Yt_raw)
    rb_mse, rb_mae, rb_mse_dim, rb_mae_dim, rb_mse_h, _ = metrics(Yb_raw, Yt_raw)

    print("\n=== Raw-scale metrics on TEST (original units) ===")
    print(f"[MODEL]    MSE={rm_mse:.6f}  MAE={rm_mae:.6f}")
    print(f"[BASELINE] MSE={rb_mse:.6f}  MAE={rb_mae:.6f}")
    print("Dim order:", use_cols)
    print("[MODEL]    MSE_dim:", rm_mse_dim, "MAE_dim:", rm_mae_dim)
    print("[BASELINE] MSE_dim:", rb_mse_dim, "MAE_dim:", rb_mae_dim)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skip plots. (pip install matplotlib)")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    sample_ids = [0, len(X_test) // 3, 2 * len(X_test) // 3]
    times_hist = np.arange(-T, 0) * dt
    times_fut = np.arange(1, H + 1) * dt

    for sid in sample_ids:
        x_hist = X_test[sid] * std + mu
        y_true = Y_test[sid] * std + mu
        y_pred = Yp[sid] * std + mu

        for d, name in enumerate(use_cols):
            plt.figure()
            plt.plot(times_hist, x_hist[:, d], label="history")
            plt.plot(times_fut, y_true[:, d], label="true")
            plt.plot(times_fut, y_pred[:, d], label="pred")
            plt.title(f"TCN+LSTM | sample {sid} | {name}")
            plt.xlabel("relative time (s)")
            plt.ylabel(name)
            plt.grid(alpha=0.3, linestyle="--")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"sample{sid}_{name}.png"), dpi=150)
            plt.close()

    plt.figure()
    plt.plot(np.arange(1, H + 1), rm_mse_h, label="model")
    plt.plot(np.arange(1, H + 1), rb_mse_h, label="baseline")
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


if __name__ == "__main__":
    main()
