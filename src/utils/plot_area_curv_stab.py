from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compute_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite_mask]
    y_f = y[finite_mask]
    n = len(x_f)

    if n < 2:
        return float("nan"), float("nan"), n

    pearson = float(np.corrcoef(x_f, y_f)[0, 1])
    spearman = compute_spearman(x_f, y_f)
    return pearson, spearman, n


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def plot_scatter(df: pd.DataFrame, y_col: str, out_png: Path) -> tuple[float, float, int]:
    x = df["iou"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite_mask]
    y_f = y[finite_mask]

    pearson, spearman, n = compute_corr(x, y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_f, y_f, alpha=0.75, s=22)
    plt.xlabel("iou")
    plt.ylabel(y_col)
    plt.title(
        f"IoU vs {y_col}\n"
        f"Pearson={pearson:.3f}, Spearman={spearman:.3f}, n={n}"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

    return pearson, spearman, n


def select_tau_columns(df: pd.DataFrame, tau_prefix: str, include_base: bool) -> list[str]:
    tau_cols: list[str] = []
    if include_base and "tau_stab" in df.columns:
        tau_cols.append("tau_stab")

    tau_cols.extend(
        col for col in df.columns if col.startswith(tau_prefix) and col not in tau_cols
    )
    return tau_cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot only tau_stab scatter(s) from area_curv_stab summary.csv"
    )
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to summary.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="all_tau",
        choices=["single", "all_tau"],
        help=(
            "single: plot only one tau column from --y_col\n"
            "all_tau: plot tau_stab plus all tau sweep columns"
        ),
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default="tau_stab",
        help="Tau column to plot in single mode",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Optional output png path in single mode",
    )
    parser.add_argument(
        "--tau_prefix",
        type=str,
        default="tau_stab__",
        help="Prefix used by tau sweep columns",
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Include the base tau_stab column in all_tau mode",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if "iou" not in df.columns:
        raise ValueError("'iou' column not found in summary.csv")

    out_root = summary_path.parent / "stab_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    corr_rows: list[dict[str, object]] = []

    if args.mode == "single":
        if args.y_col not in df.columns:
            raise ValueError(f"Tau column not found in summary.csv: {args.y_col}")

        out_png = (
            Path(args.out_png)
            if args.out_png is not None
            else out_root / f"iou_vs_{sanitize_name(args.y_col)}.png"
        )
        pearson, spearman, n = plot_scatter(df, args.y_col, out_png)
        corr_rows.append(
            {
                "x_col": "iou",
                "y_col": args.y_col,
                "pearson": pearson,
                "spearman": spearman,
                "n_finite": n,
                "out_png": str(out_png),
            }
        )
    else:
        tau_cols = select_tau_columns(
            df,
            tau_prefix=args.tau_prefix,
            include_base=args.include_base or args.tau_prefix == "tau_stab",
        )
        if not tau_cols:
            raise ValueError(
                "No tau_stab columns found. Expected 'tau_stab' and/or columns starting with "
                f"'{args.tau_prefix}'."
            )

        for y_col in tau_cols:
            out_png = out_root / f"iou_vs_{sanitize_name(y_col)}.png"
            pearson, spearman, n = plot_scatter(df, y_col, out_png)
            corr_rows.append(
                {
                    "x_col": "iou",
                    "y_col": y_col,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n_finite": n,
                    "out_png": str(out_png),
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = out_root / f"correlations_{args.mode}.csv"
    corr_df.to_csv(corr_csv, index=False)

    print(f"Saved correlation table: {corr_csv}")
    print(f"Generated {len(corr_df)} plot(s).")
    if len(corr_df) > 0:
        print(corr_df.sort_values("spearman", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
