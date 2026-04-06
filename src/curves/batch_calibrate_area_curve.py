from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_AREA_OVERRIDES = [
    "data.trainval_split=t1000",
    "data.test_split=null",
    "data.num_workers=0",
    "alpha_max=0.9",
    "alpha_steps=51",
    "w=3",
    "epsilon=0.005",
    "rho=0.1",
    "max_images=100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run calibration and area-curve analysis over training checkpoints."
    )
    parser.add_argument(
        "--train-ckpt",
        action="append",
        default=[],
        help="Explicit training checkpoint path. Repeat to pass multiple checkpoints.",
    )
    parser.add_argument(
        "--train-ckpt-glob",
        default="logs/train/runs/train_coco-person_t1000/*/checkpoints/last.ckpt",
        help="Glob used when --train-ckpt is not provided.",
    )
    parser.add_argument(
        "--calibrate-experiment",
        default="cal_coco-person_t1000_c2000_pixel",
        help="Hydra experiment name for src/calibrate.py.",
    )
    parser.add_argument(
        "--calibrate-override",
        action="append",
        default=[],
        help="Additional Hydra override for calibration. Repeat as needed.",
    )
    parser.add_argument(
        "--area-override",
        action="append",
        default=[],
        help="Additional Hydra override for area_curve.py. Repeat as needed.",
    )
    parser.add_argument(
        "--scatter-y-col",
        action="append",
        default=["tau_stab", "auc_area_ratio"],
        help="Metric to plot against IoU from summary.csv. Repeat as needed.",
    )
    parser.add_argument(
        "--output-root",
        default="logs/batch_calibrate_area_curve",
        help="Directory where per-checkpoint outputs and the manifest CSV are written.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the child jobs.",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT"),
        help="PROJECT_ROOT passed to child jobs.",
    )
    parser.add_argument(
        "--trainval-root",
        default=os.environ.get("DATA_TRAINVAL_ROOT"),
        help="DATA_TRAINVAL_ROOT passed to child jobs.",
    )
    parser.add_argument(
        "--test-root",
        default=os.environ.get("DATA_TEST_ROOT"),
        help="DATA_TEST_ROOT passed to child jobs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun calibration/area-curve even when outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def ensure_env(args: argparse.Namespace) -> dict[str, str]:
    missing = []
    if not args.project_root:
        missing.append("PROJECT_ROOT / --project-root")
    if not args.trainval_root:
        missing.append("DATA_TRAINVAL_ROOT / --trainval-root")
    if not args.test_root:
        missing.append("DATA_TEST_ROOT / --test-root")
    if missing:
        raise ValueError("Missing required paths: " + ", ".join(missing))

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(Path(args.project_root).resolve())
    env["DATA_TRAINVAL_ROOT"] = str(Path(args.trainval_root).resolve())
    env["DATA_TEST_ROOT"] = str(Path(args.test_root).resolve())
    return env


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.train_ckpt:
        checkpoints = [Path(p).expanduser().resolve() for p in args.train_ckpt]
    else:
        checkpoints = sorted(Path().glob(args.train_ckpt_glob))
        checkpoints = [p.resolve() for p in checkpoints]

    if not checkpoints:
        raise FileNotFoundError("No training checkpoints found.")

    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise FileNotFoundError("These checkpoints do not exist:\n" + "\n".join(missing))

    return checkpoints


def checkpoint_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.parts
    if "runs" in parts:
        run_idx = parts.index("runs")
        suffix = parts[run_idx + 1 :]
        if len(suffix) >= 3:
            run_name = suffix[0]
            run_stamp = suffix[1]
            return f"{run_name}__{run_stamp}"
    return ckpt_path.stem


def run_command(cmd: Sequence[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def finite_mean(values: Iterable[float]) -> float:
    cleaned = [v for v in values if math.isfinite(v)]
    if not cleaned:
        return float("nan")
    return float(sum(cleaned) / len(cleaned))


def summarize_area_curve(summary_csv: Path) -> dict[str, float]:
    rows = []
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def values(col: str) -> list[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row[col]))
            except (KeyError, TypeError, ValueError):
                out.append(float("nan"))
        return out

    return {
        "num_images": len(rows),
        "mean_iou": finite_mean(values("iou")),
        "mean_tau_stab": finite_mean(values("tau_stab")),
        "mean_auc_area_ratio": finite_mean(values("auc_area_ratio")),
        "mean_alpha_at_10_shrink": finite_mean(values("alpha_at_10_shrink")),
        "mean_alpha_at_20_shrink": finite_mean(values("alpha_at_20_shrink")),
        "mean_alpha_at_50_shrink": finite_mean(values("alpha_at_50_shrink")),
    }


def main() -> None:
    args = parse_args()
    env = ensure_env(args)
    checkpoints = discover_checkpoints(args)

    repo_root = Path(env["PROJECT_ROOT"])
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.csv"
    manifest_rows: list[dict[str, object]] = []

    area_overrides = DEFAULT_AREA_OVERRIDES + list(args.area_override)

    for train_ckpt in checkpoints:
        tag = checkpoint_tag(train_ckpt)
        run_root = output_root / tag
        calibrate_dir = run_root / "calibrate"
        area_dir = run_root / "area_curve"
        cmodel_path = calibrate_dir / "cmodel.ckpt"
        summary_csv = area_dir / "area_curves" / "summary.csv"

        calibrate_cmd = [
            args.python_bin,
            "src/calibrate.py",
            f"experiment={args.calibrate_experiment}",
            f"ckpt_path={train_ckpt}",
            f"hydra.run.dir={calibrate_dir}",
            *args.calibrate_override,
        ]

        area_cmd = [
            args.python_bin,
            "src/area_curve.py",
            f"ckpt_path={cmodel_path}",
            f"hydra.run.dir={area_dir}",
            *area_overrides,
        ]

        if args.force or not cmodel_path.exists():
            run_command(calibrate_cmd, env=env, dry_run=args.dry_run)
        else:
            print(f"Skipping calibration for {tag}; found {cmodel_path}", flush=True)

        if args.force or not summary_csv.exists():
            run_command(area_cmd, env=env, dry_run=args.dry_run)
        else:
            print(f"Skipping area_curve for {tag}; found {summary_csv}", flush=True)

        scatter_paths: list[Path] = []
        for y_col in args.scatter_y_col:
            out_png = area_dir / "area_curves" / f"iou_vs_{y_col}.png"
            scatter_cmd = [
                args.python_bin,
                "src/utils/plot_summary_scatter.py",
                "--summary_csv",
                str(summary_csv),
                "--x_col",
                "iou",
                "--y_col",
                y_col,
                "--out_png",
                str(out_png),
            ]
            if args.force or not out_png.exists():
                run_command(scatter_cmd, env=env, dry_run=args.dry_run)
            else:
                print(f"Skipping scatter for {tag}; found {out_png}", flush=True)
            scatter_paths.append(out_png)

        summary_stats = (
            summarize_area_curve(summary_csv)
            if summary_csv.exists() and not args.dry_run
            else {
                "num_images": float("nan"),
                "mean_iou": float("nan"),
                "mean_tau_stab": float("nan"),
                "mean_auc_area_ratio": float("nan"),
                "mean_alpha_at_10_shrink": float("nan"),
                "mean_alpha_at_20_shrink": float("nan"),
                "mean_alpha_at_50_shrink": float("nan"),
            }
        )

        manifest_rows.append(
            {
                "checkpoint_tag": tag,
                "train_ckpt": str(train_ckpt),
                "cmodel_ckpt": str(cmodel_path),
                "summary_csv": str(summary_csv),
                "scatter_pngs": "|".join(str(p) for p in scatter_paths),
                **summary_stats,
            }
        )

    fieldnames = [
        "checkpoint_tag",
        "train_ckpt",
        "cmodel_ckpt",
        "summary_csv",
        "scatter_pngs",
        "num_images",
        "mean_iou",
        "mean_tau_stab",
        "mean_auc_area_ratio",
        "mean_alpha_at_10_shrink",
        "mean_alpha_at_20_shrink",
        "mean_alpha_at_50_shrink",
    ]

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
