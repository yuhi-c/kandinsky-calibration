from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman = Pearson(rank(x), rank(y))
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scatter from summary.csv")
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to summary.csv")
    parser.add_argument(
        "--x_col",
        type=str,
        default="iou",
        help="Column to use for x-axis (default: iou)",
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default="tau_stab",
        help="Column to use for y-axis (default: tau_stab)",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Output png path (default: same dir as summary.csv / <x_col>_vs_<y_col>.png)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if args.x_col not in df.columns:
        raise ValueError(f"x_col not found in summary.csv: {args.x_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"y_col not found in summary.csv: {args.y_col}")

    df_plot = df.copy()

    x = df_plot[args.x_col].to_numpy(dtype=float)
    y = df_plot[args.y_col].to_numpy(dtype=float)

    # Correlations on finite pairs only
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite_mask]
    y_f = y[finite_mask]

    pearson = float(np.corrcoef(x_f, y_f)[0, 1]) if len(x_f) >= 2 else float("nan")
    spearman = compute_spearman(x_f, y_f) if len(x_f) >= 2 else float("nan")

    out_png = (
        Path(args.out_png)
        if args.out_png is not None
        else summary_path.parent / f"{args.x_col}_vs_{args.y_col}.png"
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(x_f, y_f, alpha=0.75, s=22)
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.title(
        f"{args.x_col} vs {args.y_col}\n"
        f"Pearson={pearson:.3f}, Spearman={spearman:.3f}, n={len(x_f)}"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)

    print(f"Saved plot: {out_png}")
    print(f"n_finite={len(x_f)}")
    print(f"pearson={pearson:.6f}")
    print(f"spearman={spearman:.6f}")


if __name__ == "__main__":
    main()
