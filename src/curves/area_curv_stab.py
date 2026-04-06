from __future__ import annotations

import csv
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)
    nc_curves = nc_curves.to(device)

    alphas = np.linspace(float(cfg.alpha_min), float(cfg.alpha_max), int(cfg.alpha_steps))
    if not (0.0 <= float(cfg.alpha_min) <= float(cfg.alpha_max) <= 1.0):
        raise ValueError("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1")

    fg_class_idx = int(cfg.fg_class_idx)
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
            fg_nc_curves = nc_curves[:, fg_class_idx]

            batch_size = images.shape[0]
            for b in range(batch_size):
                if max_images is not None and image_counter >= max_images:
                    break

                base_pred_mask = fg_probs[b] >= float(cfg.base_seg_threshold)
                iou = _safe_iou(base_pred_mask, fg_targets[b])
                gt_pixels = int(fg_targets[b].sum().item())

                areas = np.zeros_like(alphas, dtype=np.float64)
                for i, alpha in enumerate(alphas):
                    idx = _alpha_to_curve_index(alpha, n_curve_points)
                    thr_map = fg_nc_curves[idx]
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

    metric_dict = {
        "num_images": len(results),
        "mean_iou": float(np.nanmean([r.iou for r in results])) if results else float("nan"),
        "mean_gt_pixels": float(np.nanmean([r.gt_pixels for r in results]))
        if results
        else float("nan"),
        "mean_tau_stab": float(np.nanmean([r.tau_stab for r in results]))
        if results
        else float("nan"),
    }
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
    }
    log.info(f"Wrote tau_stab summary to: {summary_path}")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="curv/area_curv_stab.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_area_curv_stab(cfg)


if __name__ == "__main__":
    main()
