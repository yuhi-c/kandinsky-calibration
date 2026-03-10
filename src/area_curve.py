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
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@dataclass(frozen=True)
class ImageCurveResult:
    image_idx: int
    iou: float
    tau_stab: float
    auc_area_ratio: float
    alpha_at_10_shrink: float
    alpha_at_20_shrink: float
    alpha_at_50_shrink: float


def _safe_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-12) -> float:
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    intersection = torch.logical_and(pred_mask, target_mask).sum().item()
    union = torch.logical_or(pred_mask, target_mask).sum().item()
    return float(intersection / (union + eps))


def _alpha_to_curve_index(alpha: float, n_curve_points: int) -> int:
    q_level = 1.0 - float(alpha)
    idx = int(q_level * (n_curve_points - 1))
    return int(max(0, min(n_curve_points - 1, idx)))


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

    # Condition 1: area change small for w consecutive steps
    # Condition 2: meaningful shrinkage already happened
    max_k = len(diffs) - w
    for k in range(max_k + 1):
        if np.all(diffs[k : k + w] <= epsilon) and (1.0 - area_ratios[k] >= rho):
            return float(alphas[k])

    return float("nan")


def _alpha_at_shrink(alphas: np.ndarray, area_ratios: np.ndarray, shrink: float) -> float:
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink must satisfy 0 <= shrink <= 1")

    shrink_curve = 1.0 - area_ratios
    hits = np.flatnonzero(shrink_curve >= shrink)
    if len(hits) == 0:
        return float("nan")
    return float(alphas[hits[0]])


def _auc_area_ratio(alphas: np.ndarray, area_ratios: np.ndarray) -> float:
    if len(alphas) < 2:
        return float("nan")
    width = float(alphas[-1] - alphas[0])
    if width <= 0:
        return float("nan")
    # Normalize by alpha-range so values are easier to compare across sweeps.
    return float(np.trapz(area_ratios, alphas) / width)


def _plot_curve(
    *,
    out_path: Path,
    alphas: np.ndarray,
    areas: np.ndarray,
    area_ratios: np.ndarray,
    iou: float,
    tau_stab: float,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(alphas, areas)
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("|S_x(alpha)| (pixels)")
    ax[0].set_title("Prediction-set area")

    ax[1].plot(alphas, area_ratios)
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("Area ratio")
    ax[1].set_title("Normalized area ratio")

    if np.isfinite(tau_stab):
        ax[0].axvline(tau_stab, linestyle="--")
        ax[1].axvline(tau_stab, linestyle="--")

    fig.suptitle(f"IoU={iou:.3f}  tau_stab={tau_stab if np.isfinite(tau_stab) else 'nan'}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@utils.task_wrapper
def run_area_curve(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Loading checkpoint: {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    if "nc_curves" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'nc_curves'. Run calibration first (src/calibrate.py)."
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader: DataLoader = datamodule.test_dataloader()

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

    out_dir = Path(cfg.paths.output_dir) / "area_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    max_images = cfg.get("max_images")
    max_images = int(max_images) if max_images is not None else None

    results: List[ImageCurveResult] = []
    image_counter = 0

    with torch.inference_mode():
        for batch in test_loader:
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

                areas = np.zeros_like(alphas, dtype=np.float64)
                for i, alpha in enumerate(alphas):
                    idx = _alpha_to_curve_index(alpha, n_curve_points)
                    thr_map = fg_nc_curves[idx]
                    conf_mask = fg_ncs[b] <= thr_map
                    areas[i] = float(conf_mask.sum().item())

                area0 = float(areas[0])
                area_ratios = areas / (area0 + eta)

                tau = _tau_stab(
                    alphas,
                    area_ratios,
                    w=int(cfg.w),
                    epsilon=float(cfg.epsilon),
                    rho=float(cfg.rho),
                )
                auc_area_ratio = _auc_area_ratio(alphas, area_ratios)
                alpha_at_10_shrink = _alpha_at_shrink(alphas, area_ratios, 0.10)
                alpha_at_20_shrink = _alpha_at_shrink(alphas, area_ratios, 0.20)
                alpha_at_50_shrink = _alpha_at_shrink(alphas, area_ratios, 0.50)

                plot_path = curves_dir / f"image_{image_counter:05d}.png"
                _plot_curve(
                    out_path=plot_path,
                    alphas=alphas,
                    areas=areas,
                    area_ratios=area_ratios,
                    iou=iou,
                    tau_stab=tau,
                )

                results.append(
                    ImageCurveResult(
                        image_idx=image_counter,
                        iou=iou,
                        tau_stab=tau,
                        auc_area_ratio=auc_area_ratio,
                        alpha_at_10_shrink=alpha_at_10_shrink,
                        alpha_at_20_shrink=alpha_at_20_shrink,
                        alpha_at_50_shrink=alpha_at_50_shrink,
                    )
                )
                image_counter += 1

            if max_images is not None and image_counter >= max_images:
                break

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_idx",
                "iou",
                "tau_stab",
                "auc_area_ratio",
                "alpha_at_10_shrink",
                "alpha_at_20_shrink",
                "alpha_at_50_shrink",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image_idx": r.image_idx,
                    "iou": r.iou,
                    "tau_stab": r.tau_stab,
                    "auc_area_ratio": r.auc_area_ratio,
                    "alpha_at_10_shrink": r.alpha_at_10_shrink,
                    "alpha_at_20_shrink": r.alpha_at_20_shrink,
                    "alpha_at_50_shrink": r.alpha_at_50_shrink,
                }
            )

    metric_dict = {
        "num_images": len(results),
        "mean_iou": float(np.nanmean([r.iou for r in results])) if results else float("nan"),
        "mean_tau_stab": float(np.nanmean([r.tau_stab for r in results]))
        if results
        else float("nan"),
        "mean_auc_area_ratio": float(np.nanmean([r.auc_area_ratio for r in results]))
        if results
        else float("nan"),
        "mean_alpha_at_10_shrink": float(np.nanmean([r.alpha_at_10_shrink for r in results]))
        if results
        else float("nan"),
        "mean_alpha_at_20_shrink": float(np.nanmean([r.alpha_at_20_shrink for r in results]))
        if results
        else float("nan"),
        "mean_alpha_at_50_shrink": float(np.nanmean([r.alpha_at_50_shrink for r in results]))
        if results
        else float("nan"),
    }
    object_dict = {
        "cfg": cfg,
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
    }
    log.info(f"Wrote curves to: {out_dir}")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="area_curve.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_area_curve(cfg)


if __name__ == "__main__":
    main()
