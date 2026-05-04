import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CKPT_PATH, DATA_PATH, MODEL_MODE, ROOT_DIR, TRAIN_RATIO, VAL_RATIO
from dataset import load_resource_values
from model import ResourcePredictor
from utils import load_checkpoint


def make_windows(values, window_size, pred_horizon):
    xs, ys = [], []
    for i in range(window_size, len(values) - pred_horizon + 1):
        xs.append(values[i - window_size:i])
        ys.append(values[i:i + pred_horizon])
    return np.stack(xs), np.stack(ys)


def load_model(ckpt_path, device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    if cfg.get("model_mode") != MODEL_MODE:
        raise ValueError(f"checkpoint is not {MODEL_MODE}; run train.py again.")
    model = ResourcePredictor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        tcn_layers=cfg["tcn_layers"],
        lstm_hidden=cfg["lstm_hidden"],
        pred_horizon=cfg["pred_horizon"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, x_norm, device, batch_size=256):
    preds = []
    for i in range(0, len(x_norm), batch_size):
        x = torch.from_numpy(x_norm[i:i + batch_size]).to(device)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def plot_one_sample(out_dir, columns, x_raw, y_true, y_pred, sample_idx):
    for dim, name in enumerate(columns):
        hist_x = np.arange(-len(x_raw), 0)
        fut_x = np.arange(1, len(y_true) + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(hist_x, x_raw[:, dim], label="history")
        plt.plot(fut_x, y_true[:, dim], label="ground truth")
        plt.plot(fut_x, y_pred[:, dim], label="prediction")
        plt.title(f"{name} sample {sample_idx}")
        plt.xlabel("relative step")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{sample_idx}_{name}.png", dpi=150)
        plt.close()


def plot_named_sample(out_dir, columns, x_raw, y_true, y_pred, sample_idx, tag):
    for dim, name in enumerate(columns):
        hist_x = np.arange(-len(x_raw), 0)
        fut_x = np.arange(1, len(y_true) + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(hist_x, x_raw[:, dim], label="history")
        plt.plot(fut_x, y_true[:, dim], label="ground truth")
        plt.plot(fut_x, y_pred[:, dim], label="prediction")
        plt.title(f"{tag} sample {sample_idx} | {name}")
        plt.xlabel("relative step")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_sample_{sample_idx}_{name}.png", dpi=150)
        plt.close()


def plot_error_curve(out_dir, columns, y_true_raw, y_pred_raw):
    err = y_pred_raw - y_true_raw
    rmse_by_h = np.sqrt((err ** 2).mean(axis=(0, 2)))

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(rmse_by_h) + 1), rmse_by_h, marker="o")
    plt.title("RMSE by prediction horizon")
    plt.xlabel("future step")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_by_horizon.png", dpi=150)
    plt.close()

    mae_dim = np.abs(err).mean(axis=(0, 1))
    plt.figure(figsize=(7, 4))
    plt.bar(columns, mae_dim)
    plt.title("MAE by resource")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_resource.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DATA_PATH))
    parser.add_argument("--ckpt", default=str(CKPT_PATH))
    parser.add_argument("--out_dir", default=str(ROOT_DIR / "plots_eval"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    cfg = ckpt["config"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    _, values, columns = load_resource_values(args.csv, ckpt["columns"])
    n = len(values)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_raw = values[:i_train]
    val_raw = values[i_train:i_val]
    test_raw = values[i_val:]
    val_ctx = np.concatenate([train_raw[-cfg["window_size"]:], val_raw], axis=0)
    test_ctx = np.concatenate([val_ctx[-cfg["window_size"]:], test_raw], axis=0)

    x_raw, y_true_raw = make_windows(test_ctx, cfg["window_size"], cfg["pred_horizon"])
    x_norm = ((x_raw - mean) / std).astype(np.float32)
    y_pred_norm = predict_all(model, x_norm, device)
    y_pred_raw = y_pred_norm * std + mean

    err = y_pred_raw - y_true_raw
    sample_rmse = np.sqrt((err ** 2).mean(axis=(1, 2)))
    mae_all = float(np.abs(err).mean())
    rmse_all = float(np.sqrt((err ** 2).mean()))
    mae_dim = np.abs(err).mean(axis=(0, 1))
    rmse_dim = np.sqrt((err ** 2).mean(axis=(0, 1)))

    print("Using device:", device)
    print(f"TEST MAE={mae_all:.6f}, RMSE={rmse_all:.6f}")
    for name, mae, rmse in zip(columns, mae_dim, rmse_dim):
        print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

    best_idx = int(np.argmin(sample_rmse))
    worst_idx = int(np.argmax(sample_rmse))
    median_idx = int(np.argsort(sample_rmse)[len(sample_rmse) // 2])
    print(f"best sample: idx={best_idx}, RMSE={sample_rmse[best_idx]:.6f}")
    print(f"median sample: idx={median_idx}, RMSE={sample_rmse[median_idx]:.6f}")
    print(f"worst sample: idx={worst_idx}, RMSE={sample_rmse[worst_idx]:.6f}")
    for name, mae, rmse in zip(
        columns,
        np.abs(err[worst_idx]).mean(axis=0),
        np.sqrt((err[worst_idx] ** 2).mean(axis=0)),
    ):
        print(f"worst {name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

    for q in [50, 75, 90, 95, 99, 100]:
        print(f"sample RMSE P{q}: {np.percentile(sample_rmse, q):.6f}")

    sq = sample_rmse ** 2
    order = np.argsort(sq)[::-1]
    for k in [1, 5, 10, 20, 50]:
        k = min(k, len(order))
        share = sq[order[:k]].sum() / sq.sum()
        print(f"top {k} samples contribute {share * 100:.2f}% of total squared error")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = min(max(args.sample, 0), len(x_raw) - 1)
    plot_one_sample(out_dir, columns, x_raw[sample_idx], y_true_raw[sample_idx], y_pred_raw[sample_idx], sample_idx)
    plot_named_sample(out_dir, columns, x_raw[best_idx], y_true_raw[best_idx], y_pred_raw[best_idx], best_idx, "best")
    plot_named_sample(out_dir, columns, x_raw[median_idx], y_true_raw[median_idx], y_pred_raw[median_idx], median_idx, "median")
    plot_named_sample(out_dir, columns, x_raw[worst_idx], y_true_raw[worst_idx], y_pred_raw[worst_idx], worst_idx, "worst")
    plot_error_curve(out_dir, columns, y_true_raw, y_pred_raw)
    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
