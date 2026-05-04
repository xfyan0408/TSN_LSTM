import os
import numpy as np
import pandas as pd


def gen_synth(duration_s=60, dt_ms=100, seed=42, noise_std=0.15):
    """Generate smooth synthetic 3D series: Bandwidth, CPU, MEM."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * 1000 / dt_ms)
    t = np.arange(n, dtype=np.float32) * dt_ms / 1000.0  # seconds

    # Smooth bandwidth: keep 1s cycle, but add slow trend so long-term change is visible.
    bw_base = 45.0
    bw_wave_1s = 4.0 * np.sin(2.0 * np.pi * t / 1.0)
    bw_wave_slow = 6.0 * np.sin(2.0 * np.pi * t / 240.0 + 0.8)
    bw_drift = 0.004 * t
    bw_noise = rng.normal(0.0, noise_std, size=n)
    bandwidth = bw_base + bw_wave_1s + bw_wave_slow + bw_drift + bw_noise
    bandwidth = np.clip(bandwidth, 0.0, None)

    # MEM: slower variable driven by bandwidth.
    mem = np.zeros(n, dtype=np.float32)
    # Initialize near steady-state to avoid startup jump in plots.
    mem0 = 50.0 + (0.010 / 0.018) * float(bandwidth[0])
    mem[0] = float(np.clip(mem0, 0.0, 100.0))
    for i in range(1, n):
        mem[i] = (
            mem[i - 1]
            + 0.010 * bandwidth[i]
            - 0.018 * (mem[i - 1] - 50.0)
            # + rng.normal(0.0, 0.06)
        )
    mem = np.clip(mem, 0.0, 100.0)

    # CPU: faster response with 1s lag and MEM pressure.
    cpu = np.zeros(n, dtype=np.float32)
    mem_pressure0 = max(0.0, (mem[0] - 80.0) / 20.0)
    cpu0 = 8.0 + 1.15 * float(bandwidth[0]) + 22.0 * mem_pressure0
    cpu[0] = float(np.clip(cpu0, 0.0, 100.0))
    lag = int(1.0 / (dt_ms / 1000.0))  # 1s lag
    for i in range(1, n):
        j = max(0, i - lag)
        mem_pressure = max(0.0, (mem[i] - 80.0) / 20.0)
        target = 8.0 + 1.15 * bandwidth[j] + 22.0 * mem_pressure + rng.normal(0.0, 0.8)
        cpu[i] = 0.88 * cpu[i - 1] + 0.12 * target
    cpu = np.clip(cpu, 0.0, 100.0)

    start = pd.Timestamp("2026-02-19 00:00:00")
    ts = start + pd.to_timedelta(np.arange(n) * dt_ms, unit="ms")

    return pd.DataFrame({"ts": ts, "Bandwidth": bandwidth, "CPU": cpu, "MEM": mem})


def acf1(x):
    x = np.asarray(x)
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def save_quick_plot(df_100ms, df_1s, out_png="plots_data/synth_series.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plotting. (pip install matplotlib)")
        return

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    cols = ["Bandwidth", "CPU", "MEM"]

    for ax, c in zip(axes, cols):
        trend_30s = df_1s[c].rolling(window=30, min_periods=1).mean()
        ax.plot(df_100ms["ts"], df_100ms[c], label=f"{c} 100ms", linewidth=0.8, alpha=0.25)
        ax.plot(df_1s["ts"], df_1s[c], label=f"{c} 1s mean", linewidth=1.3, alpha=0.9)
        ax.plot(df_1s["ts"], trend_30s, label=f"{c} 30s trend", linewidth=2.0, color="black")
        ax.set_ylabel(c)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved plot:", out_png)


if __name__ == "__main__":
    df_100ms = gen_synth(duration_s=1700, dt_ms=100, seed=42)
    df_100ms.to_csv("data_100ms.csv", index=False)
    print("saved:", "data_100ms.csv", df_100ms.shape)

    df_1s = df_100ms.set_index("ts").resample("1s").mean().reset_index()
    df_1s.to_csv("data_1s.csv", index=False)
    print("saved:", "data_1s.csv", df_1s.shape)

    print("acf1 Bandwidth:", acf1(df_100ms["Bandwidth"]))
    print("acf1 CPU:", acf1(df_100ms["CPU"]))
    print("acf1 MEM:", acf1(df_100ms["MEM"]))

    save_quick_plot(df_100ms, df_1s)
