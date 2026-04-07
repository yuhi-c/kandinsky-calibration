from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.curves.area_curve import (
    _alpha_to_curve_index,
    _build_eval_loader,
    _build_tau_stab_specs,
    _safe_iou,
)

log = utils.get_pylogger(__name__)

TAU_SWEEP_PATTERN = re.compile(
    r"^tau_stab__w_(?P<w>[^_]+)__eps_(?P<epsilon>[^_]+)__rho_(?P<rho>[^_]+)$"
)


@dataclass(frozen=True)
class ImageTauStabResult:
    image_idx: int
    iou: float
    gt_pixels: int
    tau_stab: float
    extra_metrics: Dict[str, float]


def _tau_stab(
    alphas: np.ndarray,
    area_ratios: np.ndarray,
    *,
    w: int,
    epsilon: float,
    rho: float,
) -> float:
    if len(alphas) != len(area_ratios):
        raise ValueError("alphas and area_ratios must have same length")
    if len(alphas) < 2:
        return float("nan")
    if w <= 0:
        raise ValueError("w must be >= 1")

    diffs = np.abs(np.diff(area_ratios))
    max_k = len(diffs) - w
    for k in range(max_k + 1):
        if np.all(diffs[k : k + w] <= epsilon) and (1.0 - area_ratios[k] >= rho):
            return float(alphas[k])

    return float("nan")


def _decode_metric_token(token: str) -> float | int | str:
    if token.isdigit():
        return int(token)

    normalized = token.replace("m", "-").replace("p", ".")
    try:
        value = float(normalized)
    except ValueError:
        return normalized

    if value.is_integer():
        return int(value)
    return value


def _parse_tau_column(y_col: str) -> dict[str, float | int | str]:
    if y_col == "tau_stab":
        return {"y_col": y_col}

    match = TAU_SWEEP_PATTERN.match(y_col)
    if not match:
        return {"y_col": y_col}

    parsed = {name: _decode_metric_token(value) for name, value in match.groupdict().items()}
    parsed["y_col"] = y_col
    return parsed


def _compute_pearson(x_vals: list[float], y_vals: list[float]) -> tuple[float, int]:
    paired = [
        (x, y)
        for x, y in zip(x_vals, y_vals)
        if math.isfinite(x) and math.isfinite(y)
    ]
    n = len(paired)

    if n < 2:
        return float("nan"), n

    x_f = [x for x, _ in paired]
    y_f = [y for _, y in paired]

    mean_x = sum(x_f) / n
    mean_y = sum(y_f) / n
    sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    sum_xx = sum((x - mean_x) ** 2 for x in x_f)
    sum_yy = sum((y - mean_y) ** 2 for y in y_f)

    if sum_xx <= 0.0 or sum_yy <= 0.0:
        return float("nan"), n

    return float(sum_xy / math.sqrt(sum_xx * sum_yy)), n


def _build_pearson_rows(
    summary_rows: list[dict[str, str]],
    *,
    tau_prefix: str,
    base_params: dict[str, float | int],
    include_base: bool = True,
) -> list[dict[str, object]]:
    if not summary_rows:
        raise ValueError("summary.csv is empty")

    fieldnames = list(summary_rows[0].keys())
    if "iou" not in fieldnames:
        raise ValueError("'iou' column not found in summary.csv")

    tau_cols: list[str] = []
    if include_base and "tau_stab" in fieldnames:
        tau_cols.append("tau_stab")
    tau_cols.extend(col for col in fieldnames if col.startswith(tau_prefix))

    if not tau_cols:
        raise ValueError(
            "No tau_stab columns found. Expected 'tau_stab' and/or columns starting with "
            f"'{tau_prefix}'."
        )

    x_vals = [float(row["iou"]) for row in summary_rows]
    rows: list[dict[str, object]] = []
    for y_col in tau_cols:
        y_vals = [float(row[y_col]) for row in summary_rows]
        pearson, n = _compute_pearson(x_vals, y_vals)
        row = _parse_tau_column(y_col)
        if y_col == "tau_stab":
            row.update(base_params)
        row["is_base"] = y_col == "tau_stab"
        row["pearson"] = pearson
        row["n_finite"] = n
        rows.append(row)

    return rows


def _sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def _finite_pearson_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if math.isfinite(float(row["pearson"]))]


def _best_pearson_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    finite_rows = _finite_pearson_rows(rows)
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: float(row["pearson"]))


def _summarize_group(
    rows: list[dict[str, object]],
    *,
    highparam_name: str,
    highparam_value: object,
) -> dict[str, object]:
    finite_rows = _finite_pearson_rows(rows)
    pearsons = [float(row["pearson"]) for row in finite_rows]
    best_row = _best_pearson_row(rows)

    summary_row: dict[str, object] = {
        "highparam_name": highparam_name,
        "highparam": highparam_value,
        "num_settings": len(rows),
        "num_finite": len(finite_rows),
        "mean_pearson": float(sum(pearsons) / len(pearsons)) if pearsons else float("nan"),
        "max_pearson": max(pearsons) if pearsons else float("nan"),
        "min_pearson": min(pearsons) if pearsons else float("nan"),
    }

    if best_row is None:
        summary_row.update(
            {
                "best_pearson": float("nan"),
                "best_y_col": "",
                "best_w": "",
                "best_epsilon": "",
                "best_rho": "",
            }
        )
        return summary_row

    summary_row.update(
        {
            "best_pearson": float(best_row["pearson"]),
            "best_y_col": str(best_row["y_col"]),
            "best_w": best_row.get("w", ""),
            "best_epsilon": best_row.get("epsilon", ""),
            "best_rho": best_row.get("rho", ""),
        }
    )
    return summary_row


def _build_pearson_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    param_cols = [col for col in ("w", "epsilon", "rho") if any(col in row for row in rows)]
    varying_cols = [
        col
        for col in param_cols
        if len({row[col] for row in rows if col in row}) > 1
    ]

    if not varying_cols:
        return [_summarize_group(rows, highparam_name="setting", highparam_value="all")]

    summary_rows: list[dict[str, object]] = []
    for param_col in varying_cols:
        param_values = sorted(
            {row[param_col] for row in rows if param_col in row},
            key=_sort_key,
        )
        for param_value in param_values:
            matching_rows = [row for row in rows if row.get(param_col) == param_value]
            summary_rows.append(
                _summarize_group(
                    matching_rows,
                    highparam_name=param_col,
                    highparam_value=param_value,
                )
            )

    return summary_rows


def _write_output_table(out_csv: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {out_csv}.")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""

    fieldnames = list(rows[0].keys())
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in fieldnames
    }
    header = " ".join(field.ljust(widths[field]) for field in fieldnames)
    body = [
        " ".join(str(row.get(field, "")).ljust(widths[field]) for field in fieldnames)
        for row in rows
    ]
    return "\n".join([header, *body])


@utils.task_wrapper
def run_area_curv_stab(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Loading checkpoint: {cfg.ckpt_path}")
    ckpt = utils.load_checkpoint(cfg.ckpt_path, map_location="cpu")
    if "nc_curves" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'nc_curves'. Run calibration first (src/calibrate.py)."
        )

    eval_loader, eval_source_name = _build_eval_loader(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    nc_curves: torch.Tensor = ckpt["nc_curves"]
    n_curve_points = int(nc_curves.shape[0])

    alphas = np.linspace(float(cfg.alpha_min), float(cfg.alpha_max), int(cfg.alpha_steps))
    if not (0.0 <= float(cfg.alpha_min) <= float(cfg.alpha_max) <= 1.0):
        raise ValueError("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1")

    fg_class_idx = int(cfg.fg_class_idx)
    curve_indices = np.asarray(
        [_alpha_to_curve_index(float(alpha), n_curve_points) for alpha in alphas],
        dtype=np.int64,
    )
    unique_curve_indices, alpha_curve_lookup = np.unique(curve_indices, return_inverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)
    fg_nc_curves = nc_curves.index_select(
        0,
        torch.as_tensor(unique_curve_indices, dtype=torch.long),
    )[:, fg_class_idx].to(device)
    log.info(
        "Loaded %d/%d curve points onto device for the requested alpha grid",
        len(unique_curve_indices),
        n_curve_points,
    )

    eta = float(cfg.eta)
    tau_stab_specs = _build_tau_stab_specs(cfg)
    extra_metric_names = [
        spec.metric_name for spec in tau_stab_specs if spec.metric_name != "tau_stab"
    ]

    out_dir = Path(cfg.paths.output_dir) / "area_curv_stab"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_images = cfg.get("max_images")
    max_images = int(max_images) if max_images is not None else None

    results: List[ImageTauStabResult] = []
    image_counter = 0

    with torch.inference_mode():
        for batch in eval_loader:
            images = batch["image"].to(device)
            targets = batch["target_segmentation"].to(device)

            out = model(images)
            seg_probs = torch.sigmoid(out["seg_logits"])

            fg_probs = seg_probs[:, fg_class_idx]
            fg_targets = targets[:, fg_class_idx] > 0.5
            fg_ncs = 1.0 - fg_probs

            batch_size = images.shape[0]
            for b in range(batch_size):
                if max_images is not None and image_counter >= max_images:
                    break

                base_pred_mask = fg_probs[b] >= float(cfg.base_seg_threshold)
                iou = _safe_iou(base_pred_mask, fg_targets[b])
                gt_pixels = int(fg_targets[b].sum().item())

                areas = np.zeros_like(alphas, dtype=np.float64)
                for i, curve_lookup_idx in enumerate(alpha_curve_lookup):
                    thr_map = fg_nc_curves[int(curve_lookup_idx)]
                    conf_mask = fg_ncs[b] <= thr_map
                    areas[i] = float(conf_mask.sum().item())

                area0 = float(areas[0])
                area_ratios = areas / (area0 + eta)

                tau_metrics = {
                    spec.metric_name: _tau_stab(
                        alphas,
                        area_ratios,
                        w=spec.w,
                        epsilon=spec.epsilon,
                        rho=spec.rho,
                    )
                    for spec in tau_stab_specs
                }

                results.append(
                    ImageTauStabResult(
                        image_idx=image_counter,
                        iou=iou,
                        gt_pixels=gt_pixels,
                        tau_stab=tau_metrics["tau_stab"],
                        extra_metrics={k: v for k, v in tau_metrics.items() if k != "tau_stab"},
                    )
                )
                image_counter += 1

            if max_images is not None and image_counter >= max_images:
                break

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_idx", "iou", "gt_pixels", "tau_stab", *extra_metric_names],
        )
        writer.writeheader()
        for r in results:
            row = {
                "image_idx": r.image_idx,
                "iou": r.iou,
                "gt_pixels": r.gt_pixels,
                "tau_stab": r.tau_stab,
            }
            row.update(r.extra_metrics)
            writer.writerow(row)

    with summary_path.open(newline="") as f:
        summary_rows = list(csv.DictReader(f))

    pearson_rows = _build_pearson_rows(
        summary_rows,
        tau_prefix="tau_stab__",
        base_params={
            "w": int(cfg.w),
            "epsilon": float(cfg.epsilon),
            "rho": float(cfg.rho),
        },
    )
    pearson_summary_rows = _build_pearson_summary_rows(pearson_rows)

    pearson_path = out_dir / "pearson_only.csv"
    pearson_summary_path = out_dir / "pearson_summary.csv"
    _write_output_table(pearson_path, pearson_rows)
    _write_output_table(pearson_summary_path, pearson_summary_rows)

    best_row = _best_pearson_row(pearson_rows)

    metric_dict = {
        "num_images": len(results),
        "mean_iou": float(np.nanmean([r.iou for r in results])) if results else float("nan"),
        "mean_gt_pixels": float(np.nanmean([r.gt_pixels for r in results]))
        if results
        else float("nan"),
        "mean_tau_stab": float(np.nanmean([r.tau_stab for r in results]))
        if results
        else float("nan"),
        "num_pearson_settings": len(pearson_rows),
        "best_pearson": float(best_row["pearson"]) if best_row is not None else float("nan"),
    }
    if best_row is not None:
        metric_dict.update(
            {
                "best_pearson_w": best_row.get("w", float("nan")),
                "best_pearson_epsilon": best_row.get("epsilon", float("nan")),
                "best_pearson_rho": best_row.get("rho", float("nan")),
            }
        )

    for metric_name in extra_metric_names:
        metric_dict[f"mean_{metric_name}"] = (
            float(np.nanmean([r.extra_metrics[metric_name] for r in results]))
            if results
            else float("nan")
        )

    object_dict = {
        "cfg": cfg,
        "eval_source": eval_source_name,
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "pearson_path": str(pearson_path),
        "pearson_summary_path": str(pearson_summary_path),
    }
    log.info(f"Wrote tau_stab summary to: {summary_path}")
    log.info(f"Wrote Pearson table to: {pearson_path}")
    log.info(f"Wrote Pearson summary to: {pearson_summary_path}")

    preview_rows = sorted(
        pearson_rows,
        key=lambda row: (
            not math.isfinite(float(row["pearson"])),
            -float(row["pearson"]) if math.isfinite(float(row["pearson"])) else 0.0,
        ),
    )[:10]
    if preview_rows:
        log.info("Top Pearson settings:\n%s", _render_table(preview_rows))
    if pearson_summary_rows:
        log.info("Pearson summary:\n%s", _render_table(pearson_summary_rows))

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="curv/area_curv_stab.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_area_curv_stab(cfg)


if __name__ == "__main__":
    main()
