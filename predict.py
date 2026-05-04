import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

    def forward(self, x):  # x: (B,T,3)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]                 # (B,hidden)
        y = self.head(h_last)          # (B,H*out_dim)
        return y.view(-1, self.H, self.out_dim)  # (B,H,3)


def infer_dt_seconds(ts: pd.Series) -> float:
    """用中位数估计采样间隔（秒），避免偶发抖动影响。"""
    d = ts.diff().dropna().dt.total_seconds().to_numpy()
    if len(d) == 0:
        return 0.1
    dt = float(np.median(d))
    # 防止奇怪值
    if dt <= 0 or dt > 10:
        dt = 0.1
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_100ms.csv", help="输入时间序列CSV")
    ap.add_argument("--ckpt", default="best_lstm.pt", help="模型checkpoint")
    ap.add_argument("--out", default="pred.csv", help="输出预测CSV")
    ap.add_argument("--device", default="", help="cuda / cpu / 留空自动")
    ap.add_argument("--hidden", type=int, default=64, help="必须与训练一致")
    ap.add_argument("--num_layers", type=int, default=1, help="必须与训练一致")
    args = ap.parse_args()

    # device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读checkpoint（如果你自己生成的文件，可信；若遇到weights_only限制可fallback）
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except Exception:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    T = int(ckpt.get("T", 40))
    H = int(ckpt.get("H", 10))
    mu_t = ckpt["mu"].to(device).float()   # (3,)
    std_t = ckpt["std"].to(device).float() # (3,)

    # 建模并加载参数（注意 hidden/num_layers 要与训练一致）
    model = LSTMForecaster(in_dim=3, hidden=args.hidden, num_layers=args.num_layers, H=H, out_dim=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 读输入数据
    df = pd.read_csv(args.csv, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    # 兼容列名：Bandwidth 或 B
    if "Bandwidth" in df.columns:
        cols = ["Bandwidth", "CPU", "MEM"]
    elif "B" in df.columns:
        cols = ["B", "CPU", "MEM"]
    else:
        raise ValueError("CSV里找不到 Bandwidth 或 B 列，请检查列名。")

    if len(df) < T:
        raise ValueError(f"数据点数不足：需要至少 T={T} 个点，但当前只有 {len(df)} 个。")

    # 取最后T个点作为输入
    x_raw_np = df[cols].to_numpy(np.float32)[-T:]      # (T,3)
    x_raw = torch.from_numpy(x_raw_np).unsqueeze(0).to(device)  # (1,T,3)

    # 标准化 -> 推理 -> 反标准化
    x_norm = (x_raw - mu_t) / std_t
    with torch.no_grad():
        y_norm = model(x_norm)                         # (1,H,3)
        y_raw = y_norm * std_t + mu_t                 # (1,H,3)

    y_np = y_raw.squeeze(0).cpu().numpy()             # (H,3)

    # 构造未来时间戳
    dt = infer_dt_seconds(df["ts"])
    last_ts = df["ts"].iloc[-1]
    future_ts = [last_ts + pd.to_timedelta(dt * i, unit="s") for i in range(1, H + 1)]

    # 可选：对CPU/MEM做范围裁剪（真实项目里更建议用业务逻辑约束）
    # y_np[:,1] = np.clip(y_np[:,1], 0, 100)  # CPU
    # y_np[:,2] = np.clip(y_np[:,2], 0, 100)  # MEM
    # y_np[:,0] = np.clip(y_np[:,0], 0, None) # Bandwidth

    out = pd.DataFrame({
        "ts": future_ts,
        f"{cols[0]}_pred": y_np[:, 0],
        "CPU_pred": y_np[:, 1],
        "MEM_pred": y_np[:, 2],
    })

    out.to_csv(args.out, index=False)
    print("device:", device)
    print("T,H:", T, H, "dt(s):", dt)
    print("saved:", args.out)
    print(out.head())


if __name__ == "__main__":
    main()
