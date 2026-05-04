import argparse

import numpy as np
import pandas as pd
import torch

from models import LSTMForecaster
from scripts.common import infer_dt_seconds, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_100ms.csv", help="输入时间序列CSV")
    ap.add_argument("--ckpt", default="best_lstm.pt", help="模型checkpoint")
    ap.add_argument("--out", default="pred.csv", help="输出预测CSV")
    ap.add_argument("--device", default="", help="cuda / cpu / 留空自动")
    ap.add_argument("--hidden", type=int, default=64, help="必须与训练一致")
    ap.add_argument("--num_layers", type=int, default=1, help="必须与训练一致")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, device)

    T = int(ckpt.get("T", 40))
    H = int(ckpt.get("H", 10))
    mu_t = ckpt["mu"].to(device).float()
    std_t = ckpt["std"].to(device).float()

    model = LSTMForecaster(
        in_dim=3,
        hidden=args.hidden,
        num_layers=args.num_layers,
        H=H,
        out_dim=3,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = pd.read_csv(args.csv, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    if "Bandwidth" in df.columns:
        cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列，请检查列名。")

    if len(df) < T:
        raise ValueError(f"数据点数不足：需要至少 T={T} 个点，但当前只有 {len(df)} 个。")

    x_raw_np = df[cols].to_numpy(np.float32)[-T:]
    x_raw = torch.from_numpy(x_raw_np).unsqueeze(0).to(device)

    x_norm = (x_raw - mu_t) / std_t
    with torch.no_grad():
        y_norm = model(x_norm)
        y_raw = y_norm * std_t + mu_t

    y_np = y_raw.squeeze(0).cpu().numpy()

    dt = infer_dt_seconds(df["ts"])
    last_ts = df["ts"].iloc[-1]
    future_ts = [last_ts + pd.to_timedelta(dt * i, unit="s") for i in range(1, H + 1)]

    out = pd.DataFrame(
        {
            "ts": future_ts,
            f"{cols[0]}_pred": y_np[:, 0],
            "CPU_pred": y_np[:, 1],
            "MEM_pred": y_np[:, 2],
        }
    )
    out.to_csv(args.out, index=False)
    print("device:", device)
    print("T,H:", T, H, "dt(s):", dt)
    print("saved:", args.out)
    print(out.head())


if __name__ == "__main__":
    main()
