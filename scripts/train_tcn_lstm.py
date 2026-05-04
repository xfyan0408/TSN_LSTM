import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import TCNLSTMForecaster
from scripts.common import eval_mse, make_windows


def main():
    CSV = "data_100ms.csv"
    T, H = 40, 10
    batch_size = 256
    epochs = 30
    lr = 1e-3
    seed = 42

    model_kwargs = dict(
        in_dim=3,
        out_dim=3,
        horizon=H,
        d_model=32,
        embed_activation="relu",
        tcn_layers=5,
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4, 8, 16],
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

    df = pd.read_csv(CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    if "Bandwidth" in df.columns:
        cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列")

    series = df[cols].to_numpy(np.float32)
    n = len(series)

    i_train = int(n * 0.70)
    i_val = int(n * 0.85)

    train_raw = series[:i_train]
    val_raw = series[i_train:i_val]
    test_raw = series[i_val:]

    mu = train_raw.mean(axis=0).astype(np.float32)
    std = train_raw.std(axis=0).astype(np.float32) + 1e-6

    train = (train_raw - mu) / std
    val = (val_raw - mu) / std
    test = (test_raw - mu) / std

    X_train, Y_train = make_windows(train, T, H)
    val_ctx = np.concatenate([train[-T:], val], axis=0)
    X_val, Y_val = make_windows(val_ctx, T, H)
    test_ctx = np.concatenate([val_ctx[-T:], test], axis=0)
    X_test, Y_test = make_windows(test_ctx, T, H)

    print("X_train/Y_train:", X_train.shape, Y_train.shape)
    print("X_val/Y_val:", X_val.shape, Y_val.shape)
    print("X_test/Y_test:", X_test.shape, Y_test.shape)

    yhat = np.repeat(X_val[:, -1:, :], repeats=H, axis=1)
    baseline_mse = float(((yhat - Y_val) ** 2).mean())
    print("baseline val MSE:", baseline_mse)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNLSTMForecaster(**model_kwargs).to(device)
    print("TCN receptive field:", model.tcn.receptive_field())

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best = 1e18
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
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
                    "model_kwargs": model_kwargs,
                },
                "best_tcn_lstm.pt",
            )

    ckpt = torch.load("best_tcn_lstm.pt", map_location=device)
    best_model = TCNLSTMForecaster(**ckpt["model_kwargs"]).to(device)
    best_model.load_state_dict(ckpt["state_dict"])
    test_mse = eval_mse(best_model, test_ld, device)
    print("best val_mse:", best)
    print("test_mse:", test_mse)
    print("saved:", "best_tcn_lstm.pt")


if __name__ == "__main__":
    main()
