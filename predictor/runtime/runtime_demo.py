import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# runtime 代码在 predictor/runtime 下，需要把 predictor 目录加入 import 路径。
PREDICTOR_DIR = Path(__file__).resolve().parents[1]
if str(PREDICTOR_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_DIR))

from config import CKPT_PATH, DATA_PATH, MODEL_MODE, ROOT_DIR  # noqa: E402
from dataset import load_resource_values  # noqa: E402
from model import ResourcePredictor  # noqa: E402
from utils import infer_dt_seconds, load_checkpoint  # noqa: E402


# 运行时策略参数先集中放在这里，避免过早拆散到多个文件。
RESOURCE_THRESHOLDS = {
    "Bandwidth": 80.0,
    "CPU": 80.0,
    "MEM": 85.0,
}

TARGET_UTILIZATION = {
    "Bandwidth": 70.0,
    "CPU": 70.0,
    "MEM": 75.0,
}

CONSECUTIVE_STEPS = 2
SAMPLE_INTERVAL_SECONDS = 5.0
CURRENT_INSTANCES = 4


def load_runtime_model(ckpt_path, device):
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


def build_future_time(df, horizon, sample_interval_seconds):
    # 如果 CSV 有 ts 列，就沿用实际时间间隔；否则用 step 编号。
    if "ts" not in df.columns:
        return np.arange(1, horizon + 1)

    dt = infer_dt_seconds(df["ts"])
    if sample_interval_seconds > 0:
        dt = sample_interval_seconds

    last_ts = pd.to_datetime(df["ts"].iloc[-1])
    return [last_ts + pd.to_timedelta(dt * i, unit="s") for i in range(1, horizon + 1)]


def normalize_end_index(end_index, row_count):
    # end_index 表示历史窗口结束位置；0 表示使用 CSV 最后一行作为运行时最新点。
    if end_index <= 0:
        return row_count
    return min(end_index, row_count)


def predict_recent_window(csv_path, ckpt_path, device, sample_interval_seconds, end_index, history_points, compare_recent_points):
    model, ckpt = load_runtime_model(ckpt_path, device)
    cfg = ckpt["config"]
    mean = ckpt["mean"]
    std = ckpt["std"]
    pred_horizon = cfg["pred_horizon"]

    df, values, columns = load_resource_values(csv_path, ckpt["columns"])
    if end_index <= 0 and compare_recent_points > 0:
        if compare_recent_points < pred_horizon:
            raise ValueError(f"compare_recent_points must be >= pred_horizon({pred_horizon})")
        end_index = len(values) - compare_recent_points

    end_index = normalize_end_index(end_index, len(values))
    if end_index < history_points:
        raise ValueError(f"need end_index >= {history_points}, got {end_index}")

    # 运行时只取最近 history_points 步；回放模式下取真实未来之前的 20 步。
    recent_raw = values[end_index - history_points:end_index]
    recent_norm = (recent_raw - mean) / std
    x = torch.from_numpy(recent_norm.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x).squeeze(0).cpu().numpy()

    # 只反归一化一次，得到原始资源量纲下的未来预测值。
    pred_raw = pred_norm * std + mean
    history_df = df.iloc[:end_index]
    future_time = build_future_time(history_df, len(pred_raw), sample_interval_seconds)
    forecast = pd.DataFrame(pred_raw, columns=columns)
    forecast.insert(0, "time", future_time)
    forecast.insert(1, "step", np.arange(1, len(pred_raw) + 1))

    # 如果 CSV 后面还有真实未来 H 步，就可以计算运行时回放误差。
    truth_raw = None
    if end_index + pred_horizon <= len(values):
        truth_raw = values[end_index:end_index + pred_horizon]
        if "ts" in df.columns:
            forecast["time"] = pd.to_datetime(df["ts"].iloc[end_index:end_index + pred_horizon]).to_list()

    return forecast, columns, truth_raw, end_index


def compute_error_report(forecast, columns, truth_raw):
    if truth_raw is None:
        return None

    pred = forecast[list(columns)].to_numpy(dtype=np.float32)
    err = pred - truth_raw
    mae_dim = np.abs(err).mean(axis=0)
    rmse_dim = np.sqrt((err ** 2).mean(axis=0))
    mape_dim = (np.abs(err) / np.maximum(np.abs(truth_raw), 1e-8)).mean(axis=0) * 100

    return {
        "mae_all": float(np.abs(err).mean()),
        "rmse_all": float(np.sqrt((err ** 2).mean())),
        "mape_all": float((np.abs(err) / np.maximum(np.abs(truth_raw), 1e-8)).mean() * 100),
        "mae_dim": mae_dim,
        "rmse_dim": rmse_dim,
        "mape_dim": mape_dim,
        "truth_raw": truth_raw,
    }


def first_consecutive_violation(values, threshold, consecutive_steps):
    # 找到第一次连续 N 步超过阈值的位置，返回起始 step 下标。
    run = 0
    for idx, value in enumerate(values):
        if value > threshold:
            run += 1
            if run >= consecutive_steps:
                return idx - consecutive_steps + 1
        else:
            run = 0
    return None


def make_decision(forecast, columns, current_instances):
    pred = forecast[list(columns)].to_numpy(dtype=np.float32)
    violations = []

    for dim, name in enumerate(columns):
        threshold = RESOURCE_THRESHOLDS.get(name, 80.0)
        first_idx = first_consecutive_violation(pred[:, dim], threshold, CONSECUTIVE_STEPS)
        if first_idx is not None:
            max_value = float(pred[:, dim].max())
            violations.append(
                {
                    "resource": name,
                    "first_step": int(forecast["step"].iloc[first_idx]),
                    "first_time": forecast["time"].iloc[first_idx],
                    "threshold": threshold,
                    "max_pred": max_value,
                    "excess_ratio": max_value / threshold,
                }
            )

    if not violations:
        return {
            "status": "SAFE",
            "bottleneck": "-",
            "first_step": None,
            "first_time": None,
            "suggested_instances": current_instances,
            "reason": "预测窗口内没有资源连续超过阈值。",
        }

    # 先按最早越界排序；同一步越界时，选超过阈值比例最大的资源。
    violations.sort(key=lambda item: (item["first_step"], -item["excess_ratio"]))
    bottleneck = violations[0]

    scale_factor = 1.0
    for dim, name in enumerate(columns):
        target = TARGET_UTILIZATION.get(name, 70.0)
        max_pred = float(pred[:, dim].max())
        scale_factor = max(scale_factor, max_pred / target)

    suggested_instances = max(current_instances + 1, math.ceil(current_instances * scale_factor))
    return {
        "status": "SCALE_OUT_REQUIRED",
        "bottleneck": bottleneck["resource"],
        "first_step": bottleneck["first_step"],
        "first_time": bottleneck["first_time"],
        "threshold": bottleneck["threshold"],
        "max_pred": bottleneck["max_pred"],
        "suggested_instances": suggested_instances,
        "reason": f"{bottleneck['resource']} 预计连续 {CONSECUTIVE_STEPS} 步超过阈值。",
    }


def add_step_status(forecast, columns, truth_raw=None):
    # 给预测表加一列状态，方便直接看每一步是否触发风险。
    statuses = []
    for _, row in forecast.iterrows():
        warnings = []
        for name in columns:
            if row[name] > RESOURCE_THRESHOLDS.get(name, 80.0):
                warnings.append(f"{name} warning")
        statuses.append("safe" if not warnings else ", ".join(warnings))

    result = forecast.copy()
    result["status"] = statuses
    if truth_raw is not None:
        # 回放模式下，把真实值和逐点误差也写进表格，方便核对预测准不准。
        for dim, name in enumerate(columns):
            result[f"{name}_true"] = truth_raw[:, dim]
            result[f"{name}_abs_err"] = np.abs(result[name] - result[f"{name}_true"])
    return result


def print_error_report(columns, error_report):
    print("\nRuntime error:")
    if error_report is None:
        print("  暂无误差：当前使用的是 CSV 最新窗口，未来真实值还没有到达。")
        print("  如需回放计算误差，请使用 --end_index 指定一个历史截止位置。")
        return

    print(f"  Overall MAE={error_report['mae_all']:.6f}, RMSE={error_report['rmse_all']:.6f}, MAPE={error_report['mape_all']:.3f}%")
    print(f"  {'resource':<12} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
    for name, mae, rmse, mape in zip(columns, error_report["mae_dim"], error_report["rmse_dim"], error_report["mape_dim"]):
        print(f"  {name:<12} {mae:>12.6f} {rmse:>12.6f} {mape:>11.3f}%")


def print_report(forecast, columns, decision, current_instances, end_index, history_points, error_report):
    max_pred = forecast[list(columns)].max()

    print("\n========== Runtime Prediction ==========")
    print(f"Runtime end_index: {end_index}")
    if error_report is None:
        print("Mode: latest window, no future truth yet")
    else:
        print("Mode: replay, use history points then compare with known future truth")
    print(f"History points used: {history_points}")
    print(f"Forecast horizon: {len(forecast)} steps")
    print(f"Current instances: {current_instances}")

    print("\nMax predicted utilization:")
    for name in columns:
        print(f"  {name}: {max_pred[name]:.3f} (threshold={RESOURCE_THRESHOLDS.get(name, 80.0):.1f})")

    print("\nDecision:")
    print(f"  Status: {decision['status']}")
    print(f"  Bottleneck: {decision['bottleneck']}")
    if decision["first_step"] is not None:
        print(f"  First violation: step {decision['first_step']}, time={decision['first_time']}")
        print(f"  Max predicted {decision['bottleneck']}: {decision['max_pred']:.3f}")
        print(f"  Threshold: {decision['threshold']:.1f}")
    print(f"  Suggested instances: {current_instances} -> {decision['suggested_instances']}")
    print(f"  Reason: {decision['reason']}")
    print_error_report(columns, error_report)

    print("\nForecast table:")
    print(add_step_status(forecast, columns, None if error_report is None else error_report["truth_raw"]).to_string(index=False))


def plot_forecast(forecast, columns, decision, out_path):
    plt.figure(figsize=(9, 5))
    x = forecast["step"].to_numpy()

    for name in columns:
        plt.plot(x, forecast[name], marker="o", label=name)
        plt.axhline(RESOURCE_THRESHOLDS.get(name, 80.0), linestyle="--", linewidth=1, label=f"{name} threshold")

    plt.title(f"Runtime forecast | {decision['status']}")
    plt.xlabel("future step")
    plt.ylabel("resource utilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DATA_PATH))
    parser.add_argument("--ckpt", default=str(CKPT_PATH))
    parser.add_argument("--out_dir", default=str(ROOT_DIR / "runtime_outputs"))
    parser.add_argument("--device", default="")
    parser.add_argument("--current_instances", type=int, default=CURRENT_INSTANCES)
    parser.add_argument("--sample_interval_seconds", type=float, default=SAMPLE_INTERVAL_SECONDS)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--history_points", type=int, default=20)
    parser.add_argument("--compare_recent_points", type=int, default=10)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast, columns, truth_raw, end_index = predict_recent_window(
        csv_path=args.csv,
        ckpt_path=args.ckpt,
        device=device,
        sample_interval_seconds=args.sample_interval_seconds,
        end_index=args.end_index,
        history_points=args.history_points,
        compare_recent_points=args.compare_recent_points,
    )
    
    decision = make_decision(forecast, columns, args.current_instances)
    
    error_report = compute_error_report(forecast, columns, truth_raw)

    forecast_with_status = add_step_status(forecast, columns, truth_raw)
    forecast_with_status.to_csv(out_dir / "runtime_forecast.csv", index=False)
    plot_forecast(forecast, columns, decision, out_dir / "runtime_forecast.png")
    print_report(forecast, columns, decision, args.current_instances, end_index, args.history_points, error_report)
    print(f"\nSaved runtime outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
