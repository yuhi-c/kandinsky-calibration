from __future__ import annotations

import csv
import itertools
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
from src.data.components.coco_dataset import CocoSegmentationDataset

log = utils.get_pylogger(__name__)


@dataclass(frozen=True)
class ImageCurveResult:
    image_idx: int
    iou: float
    gt_pixels: int
    tau_stab: float
    auc_area_ratio: float
    alpha_at_10_shrink: float
    alpha_at_20_shrink: float
    alpha_at_50_shrink: float

    # new: early-slope features
    slope_0_3: float
    max_drop_first_5: float

    # new: knee features
    knee_alpha_second_diff: float
    knee_strength_second_diff: float
    extra_metrics: Dict[str, float]


@dataclass(frozen=True)
class TauStabSpec:
    metric_name: str
    w: int
    epsilon: float
    rho: float


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


def _format_metric_token(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _tau_stab_metric_name(w: int, epsilon: float, rho: float) -> str:
    return (
        f"tau_stab__w_{_format_metric_token(w)}"
        f"__eps_{_format_metric_token(epsilon)}"
        f"__rho_{_format_metric_token(rho)}"
    )


def _build_tau_stab_specs(cfg: DictConfig) -> List[TauStabSpec]:
    default_spec = TauStabSpec(
        metric_name="tau_stab",
        w=int(cfg.w),
        epsilon=float(cfg.epsilon),
        rho=float(cfg.rho),
    )

    sweep_cfg = cfg.get("tau_stab_sweep")
    if not sweep_cfg or not bool(sweep_cfg.get("enabled", False)):
        return [default_spec]

    w_values = sweep_cfg.get("w_values") or [int(cfg.w)]
    epsilon_values = sweep_cfg.get("epsilon_values") or [float(cfg.epsilon)]
    rho_values = sweep_cfg.get("rho_values") or [float(cfg.rho)]

    specs: List[TauStabSpec] = [default_spec]
    seen = {(default_spec.w, default_spec.epsilon, default_spec.rho)}
    for w, epsilon, rho in itertools.product(w_values, epsilon_values, rho_values):
        key = (int(w), float(epsilon), float(rho))
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            TauStabSpec(
                metric_name=_tau_stab_metric_name(*key),
                w=key[0],
                epsilon=key[1],
                rho=key[2],
            )
        )

    return specs


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
    return float(np.trapz(area_ratios, alphas) / width)


def _early_slope(alphas: np.ndarray, area_ratios: np.ndarray, end_idx: int = 3) -> float:
    """
    slope from alpha[0] to alpha[end_idx]
    Usually negative or zero because area ratio decreases as alpha increases.
    """
    if len(alphas) <= end_idx:
        return float("nan")
    dx = float(alphas[end_idx] - alphas[0])
    if dx <= 0:
        return float("nan")
    dy = float(area_ratios[end_idx] - area_ratios[0])
    return float(dy / dx)


def _max_drop_first_k(area_ratios: np.ndarray, k: int = 5) -> float:
    """
    maximum one-step drop in the first k transitions.
    Returns positive value (larger means steeper early shrink).
    """
    if len(area_ratios) < 2:
        return float("nan")
    diffs = area_ratios[:-1] - area_ratios[1:]
    k_eff = min(k, len(diffs))
    if k_eff <= 0:
        return float("nan")
    return float(np.max(diffs[:k_eff]))


def _knee_by_second_diff(alphas: np.ndarray, area_ratios: np.ndarray) -> tuple[float, float]:
    """
    Simple knee detector via discrete second difference.
    We use shrink_curve = 1 - area_ratios so that the "growth of shrinkage"
    is emphasized. The knee is where second diff is largest.
    Returns:
        knee_alpha, knee_strength
    """
    if len(alphas) < 3 or len(area_ratios) < 3:
        return float("nan"), float("nan")

    shrink_curve = 1.0 - area_ratios
    second = np.diff(shrink_curve, n=2)  # length n-2
    if len(second) == 0:
        return float("nan"), float("nan")

    k = int(np.argmax(second))
    # second[k] corresponds roughly to center index k+1
    center_idx = k + 1
    return float(alphas[center_idx]), float(second[k])


def _plot_curve(
    *,
    out_path: Path,
    alphas: np.ndarray,
    areas: np.ndarray,
    area_ratios: np.ndarray,
    iou: float,
    gt_pixels: int,
    tau_stab: float,
    slope_0_3: float,
    knee_alpha_second_diff: float,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(alphas, areas, label="Prediction-set area")
    ax[0].axhline(gt_pixels, linestyle=":", color="tab:red", label="GT pixels")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("|S_x(alpha)| (pixels)")
    ax[0].set_title("Prediction-set area")
    ax[0].legend()

    ax[1].plot(alphas, area_ratios, label="Area ratio")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("Area ratio")
    ax[1].set_title("Normalized area ratio")

    if np.isfinite(tau_stab):
        ax[0].axvline(tau_stab, linestyle="--", alpha=0.8, label="tau_stab")
        ax[1].axvline(tau_stab, linestyle="--", alpha=0.8, label="tau_stab")

    if np.isfinite(knee_alpha_second_diff):
        ax[0].axvline(knee_alpha_second_diff, linestyle=":", alpha=0.8, label="knee")
        ax[1].axvline(knee_alpha_second_diff, linestyle=":", alpha=0.8, label="knee")

    ax[1].legend()

    fig.suptitle(
        "  ".join(
            [
                f"IoU={iou:.3f}",
                f"GT pixels={gt_pixels}",
                f"tau_stab={tau_stab if np.isfinite(tau_stab) else 'nan'}",
                f"slope_0_3={slope_0_3 if np.isfinite(slope_0_3) else 'nan':.3f}",
                f"knee={knee_alpha_second_diff if np.isfinite(knee_alpha_second_diff) else 'nan'}",
            ]
        )
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _build_eval_loader(cfg: DictConfig) -> tuple[DataLoader, str]:
    eval_source = str(cfg.data.get("eval_source", "test"))
    eval_subset = str(cfg.data.get("eval_subset", "test"))

    if eval_source == "test":
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup(stage="test")
        return datamodule.test_dataloader(), "test"

    if eval_source != "trainval":
        raise ValueError("data.eval_source must be either 'test' or 'trainval'")

    manifest_by_subset = {
        "train": "labels_train.json",
        "val": "labels_val.json",
        "cal": "labels_cal.json",
    }
    if eval_subset not in manifest_by_subset:
        raise ValueError("data.eval_subset must be one of: train, val, cal")

    manifest_name = manifest_by_subset[eval_subset]
    manifest_fn = (
        Path(cfg.data.trainval_root) / "splits" / str(cfg.data.trainval_split) / manifest_name
    )
    log.info(f"Loading evaluation dataset from trainval split manifest: {manifest_fn}")

    dataset = CocoSegmentationDataset(
        data_dir=cfg.data.trainval_root,
        manifest_fn=manifest_fn,
        dims=tuple(cfg.data.dims),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
    )
    return dataloader, f"trainval/{cfg.data.trainval_split}/{manifest_name}"


@utils.task_wrapper
def run_area_curve(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    extra_metric_names = [spec.metric_name for spec in tau_stab_specs if spec.metric_name != "tau_stab"]

    out_dir = Path(cfg.paths.output_dir) / "area_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    max_images = cfg.get("max_images")
    max_images = int(max_images) if max_images is not None else None

    results: List[ImageCurveResult] = []
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
                tau = tau_metrics["tau_stab"]
                auc_area_ratio = _auc_area_ratio(alphas, area_ratios)
                alpha_at_10_shrink = _alpha_at_shrink(alphas, area_ratios, 0.10)
                alpha_at_20_shrink = _alpha_at_shrink(alphas, area_ratios, 0.20)
                alpha_at_50_shrink = _alpha_at_shrink(alphas, area_ratios, 0.50)

                slope_0_3 = _early_slope(alphas, area_ratios, end_idx=3)
                max_drop_first_5 = _max_drop_first_k(area_ratios, k=5)
                knee_alpha_second_diff, knee_strength_second_diff = _knee_by_second_diff(
                    alphas, area_ratios
                )

                plot_path = curves_dir / f"image_{image_counter:05d}.png"
                _plot_curve(
                    out_path=plot_path,
                    alphas=alphas,
                    areas=areas,
                    area_ratios=area_ratios,
                    iou=iou,
                    gt_pixels=gt_pixels,
                    tau_stab=tau,
                    slope_0_3=slope_0_3,
                    knee_alpha_second_diff=knee_alpha_second_diff,
                )

                results.append(
                    ImageCurveResult(
                        image_idx=image_counter,
                        iou=iou,
                        gt_pixels=gt_pixels,
                        tau_stab=tau,
                        auc_area_ratio=auc_area_ratio,
                        alpha_at_10_shrink=alpha_at_10_shrink,
                        alpha_at_20_shrink=alpha_at_20_shrink,
                        alpha_at_50_shrink=alpha_at_50_shrink,
                        slope_0_3=slope_0_3,
                        max_drop_first_5=max_drop_first_5,
                        knee_alpha_second_diff=knee_alpha_second_diff,
                        knee_strength_second_diff=knee_strength_second_diff,
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
            fieldnames=[
                "image_idx",
                "iou",
                "gt_pixels",
                "tau_stab",
                "auc_area_ratio",
                "alpha_at_10_shrink",
                "alpha_at_20_shrink",
                "alpha_at_50_shrink",
                "slope_0_3",
                "max_drop_first_5",
                "knee_alpha_second_diff",
                "knee_strength_second_diff",
                *extra_metric_names,
            ],
        )
        writer.writeheader()
        for r in results:
            row = {
                "image_idx": r.image_idx,
                "iou": r.iou,
                "gt_pixels": r.gt_pixels,
                "tau_stab": r.tau_stab,
                "auc_area_ratio": r.auc_area_ratio,
                "alpha_at_10_shrink": r.alpha_at_10_shrink,
                "alpha_at_20_shrink": r.alpha_at_20_shrink,
                "alpha_at_50_shrink": r.alpha_at_50_shrink,
                "slope_0_3": r.slope_0_3,
                "max_drop_first_5": r.max_drop_first_5,
                "knee_alpha_second_diff": r.knee_alpha_second_diff,
                "knee_strength_second_diff": r.knee_strength_second_diff,
            }
            row.update(r.extra_metrics)
            writer.writerow(row)

    metric_dict = {
        "num_images": len(results),
        "mean_iou": float(np.nanmean([r.iou for r in results])) if results else float("nan"),
        "mean_tau_stab": float(np.nanmean([r.tau_stab for r in results])) if results else float("nan"),
        "mean_auc_area_ratio": float(np.nanmean([r.auc_area_ratio for r in results])) if results else float("nan"),
        "mean_alpha_at_10_shrink": float(np.nanmean([r.alpha_at_10_shrink for r in results])) if results else float("nan"),
        "mean_alpha_at_20_shrink": float(np.nanmean([r.alpha_at_20_shrink for r in results])) if results else float("nan"),
        "mean_alpha_at_50_shrink": float(np.nanmean([r.alpha_at_50_shrink for r in results])) if results else float("nan"),
        "mean_slope_0_3": float(np.nanmean([r.slope_0_3 for r in results])) if results else float("nan"),
        "mean_max_drop_first_5": float(np.nanmean([r.max_drop_first_5 for r in results])) if results else float("nan"),
        "mean_knee_alpha_second_diff": float(np.nanmean([r.knee_alpha_second_diff for r in results])) if results else float("nan"),
        "mean_knee_strength_second_diff": float(np.nanmean([r.knee_strength_second_diff for r in results])) if results else float("nan"),
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
    log.info(f"Wrote curves to: {out_dir}")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="curv/area_curve.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_area_curve(cfg)


if __name__ == "__main__":
    main()
