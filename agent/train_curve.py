import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 避免 PyCharm / backend 显示报错

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "savefig.dpi": 600,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def smooth(series: pd.Series, window: int = 31) -> pd.Series:
    return series.rolling(
        window=window,
        center=True,
        min_periods=max(3, window // 4)
    ).mean()


def plot_fulfillment_rate(csv_path, smooth_window=31, outdir="figures", case_name="DDT1200_num100"):
    df = pd.read_csv(csv_path)

    required_cols = {"iteration", "fulfillment_rate_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df = df.sort_values("iteration")

    x = df["iteration"]
    y_raw = df["fulfillment_rate_mean"] * 100.0
    y_smooth = smooth(y_raw, smooth_window)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))

    # 背景和网格，尽量贴近你给的示例风格
    ax.set_facecolor("#EAEAF2")
    ax.grid(True, color="gray", alpha=0.45, linewidth=0.8)

    # 原始曲线（浅色）
    ax.plot(
        x, y_raw,
        color="#5DA5DA",
        linewidth=1.0,
        alpha=0.35,
        label="Fulfillment Rate"
    )

    # 移动平均曲线（深色）
    ax.plot(
        x, y_smooth,
        color="#F28E2B",
        linewidth=1.5,
        alpha=0.95,
        label="Moving Average"
    )

    # 顶部放算例名
    ax.set_title(case_name, pad=6)

    # 坐标轴标签
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fulfillment Rate (%)")

    # 图例放右边（右上角）
    ax.legend(loc="upper right", frameon=True)

    # 坐标轴范围稍微留白
    ax.set_xlim(x.min(), x.max())
    y_min = float(y_raw.min())
    y_max = float(y_raw.max())
    margin = max((y_max - y_min) * 0.08, 1.0)
    ax.set_ylim(max(0, y_min - margin), min(100, y_max + margin))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdf_path = outdir / f"{case_name}_fulfillment_rate_mean.pdf"
    png_path = outdir / f"{case_name}_fulfillment_rate_mean.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="/Users/zhuyuhao/Code/HGNN_Fluid_FFJSP-master/result/Fluid-HGAT/DDT1200_num100.txt.csv",
        help="Path to CSV file"
    )
    parser.add_argument("--smooth", type=int, default=31, help="Moving average window")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    parser.add_argument("--case_name", type=str, default="DDT1200_num100", help="Case name shown on top")
    args = parser.parse_args()

    plot_fulfillment_rate(
        csv_path=args.csv,
        smooth_window=args.smooth,
        outdir=args.outdir,
        case_name=args.case_name
    )