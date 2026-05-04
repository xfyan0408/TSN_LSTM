import argparse

import numpy as np
import pandas as pd
import torch

from config import CKPT_PATH, DATA_PATH, MODEL_MODE
from dataset import load_resource_values
from model import ResourcePredictor
from utils import infer_dt_seconds, load_checkpoint


def load_model(ckpt_path=CKPT_PATH, device="cpu"):
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


def predict_recent(csv_path=DATA_PATH, ckpt_path=CKPT_PATH, out_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(ckpt_path, device)

    df, values, columns = load_resource_values(csv_path, ckpt["columns"])
    mean = ckpt["mean"]
    std = ckpt["std"]
    window_size = ckpt["config"]["window_size"]

    if len(values) < window_size:
        raise ValueError(f"need at least {window_size} rows, got {len(values)}")

    recent = values[-window_size:]
    recent_norm = (recent - mean) / std
    x = torch.from_numpy(recent_norm.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x).squeeze(0).cpu().numpy()
    pred = pred_norm * std + mean

    if "ts" in df.columns:
        dt = infer_dt_seconds(df["ts"])
        last_ts = pd.to_datetime(df["ts"].iloc[-1])
        future_ts = [last_ts + pd.to_timedelta(dt * i, unit="s") for i in range(1, len(pred) + 1)]
    else:
        future_ts = np.arange(1, len(pred) + 1)

    result = pd.DataFrame(pred, columns=[f"{c}_pred" for c in columns])
    result.insert(0, "time", future_ts)

    if out_path:
        result.to_csv(out_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DATA_PATH))
    parser.add_argument("--ckpt", default=str(CKPT_PATH))
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    result = predict_recent(
        csv_path=args.csv,
        ckpt_path=args.ckpt,
        out_path=args.out or None,
        device=args.device or None,
    )
    print("未来资源需求预测:")
    print(result)


if __name__ == "__main__":
    main()
