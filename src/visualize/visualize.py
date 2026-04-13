from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, Subset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


def _alpha_to_curve_index(alpha: float, n_curve_points: int) -> int:
    q_level = 1.0 - float(alpha)
    idx = int(q_level * (n_curve_points - 1))
    return int(max(0, min(n_curve_points - 1, idx)))


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _resolve_companion_cfg_path(cfg: DictConfig) -> Path:
    if cfg.get("companion_cfg_path"):
        companion_cfg_path = Path(cfg.companion_cfg_path).expanduser().resolve()
    else:
        ckpt_path = Path(cfg.ckpt_path).expanduser().resolve()
        companion_cfg_path = ckpt_path.parent / ".hydra" / "config.yaml"

    if not companion_cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find companion Hydra config at: {companion_cfg_path}"
        )

    return companion_cfg_path


def _build_dataset(datamodule: Any, stage: str) -> Dataset:
    if stage == "calibrate":
        datamodule.setup(stage="calibrate")
        return datamodule.val_dataloader().dataset
    if stage == "validate":
        datamodule.setup(stage="validate")
        return datamodule.val_dataloader().dataset
    if stage == "test":
        datamodule.setup(stage="test")
        return datamodule.test_dataloader().dataset

    raise ValueError("stage must be one of: calibrate, validate, test")


def _resolve_source_index(dataset: Dataset, sample_idx: int) -> int:
    if isinstance(dataset, Subset):
        return int(dataset.indices[sample_idx])
    return int(sample_idx)


def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0.0, 1.0)


def _save_binary_mask(mask: np.ndarray, out_path: Path) -> None:
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img.save(out_path)


def _save_rgb_image(image_np: np.ndarray, out_path: Path) -> None:
    image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(out_path)


def _draw_overlay(
    ax: Any,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    title: str,
    color: Tuple[float, float, float],
    overlay_alpha: float,
) -> None:
    ax.imshow(image_np)

    overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask_np.astype(np.float32) * overlay_alpha
    ax.imshow(overlay)

    ax.set_title(title)
    ax.axis("off")


def _save_image_grid(
    *,
    out_path: Path,
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    alpha_masks: Sequence[Tuple[float, np.ndarray, int, float]],
    overlay_alpha: float,
    source_index: int,
) -> None:
    panels = 2 + len(alpha_masks)
    ncols = 3
    nrows = math.ceil(panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(image_np)
    axes[0].set_title(f"Image (idx={source_index})")
    axes[0].axis("off")

    _draw_overlay(
        axes[1],
        image_np,
        gt_mask,
        title=f"Ground truth\npixels={int(gt_mask.sum())}",
        color=(0.2, 0.85, 0.35),
        overlay_alpha=overlay_alpha,
    )

    for plot_idx, (alpha, mask_np, pixels, area_ratio) in enumerate(alpha_masks, start=2):
        _draw_overlay(
            axes[plot_idx],
            image_np,
            mask_np,
            title=f"alpha={alpha:.1f}\npixels={pixels}, ratio={area_ratio:.3f}",
            color=(1.0, 0.45, 0.1),
            overlay_alpha=overlay_alpha,
        )

    for ax in axes[panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@utils.task_wrapper
def visualize(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_path = Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    save_dir = Path(cfg.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    companion_cfg_path = _resolve_companion_cfg_path(cfg)
    run_cfg = OmegaConf.load(companion_cfg_path)

    log.info(f"Using checkpoint: {ckpt_path}")
    log.info(f"Using companion config: {companion_cfg_path}")

    log.info(f"Instantiating datamodule from calibration config <{run_cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(run_cfg.data)
    dataset = _build_dataset(datamodule, str(cfg.stage))

    log.info(f"Instantiating model from calibration config <{run_cfg.model._target_}>")
    model = hydra.utils.instantiate(run_cfg.model)

    checkpoint = utils.load_checkpoint(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    if "nc_curves" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'nc_curves'. Run calibration first.")

    fg_class_idx = int(cfg.fg_class_idx) if cfg.get("fg_class_idx") is not None else int(
        run_cfg.get("class_idx", 1)
    )

    device = _resolve_device(str(cfg.device))
    nc_curves = checkpoint["nc_curves"].to(device)
    n_curve_points = int(nc_curves.shape[0])
    fg_nc_curves = nc_curves[:, fg_class_idx]

    model = model.to(device)
    model.eval()

    alpha_values = [float(alpha) for alpha in cfg.alpha_values]
    image_indices = [int(idx) for idx in cfg.image_indices]

    summary_rows: List[Dict[str, Any]] = []
    figure_paths: List[str] = []

    with torch.inference_mode():
        for sample_idx in image_indices:
            if sample_idx < 0 or sample_idx >= len(dataset):
                raise IndexError(
                    f"image index {sample_idx} is out of range for stage '{cfg.stage}' "
                    f"(dataset size: {len(dataset)})"
                )

            sample = dataset[sample_idx]
            source_index = _resolve_source_index(dataset, sample_idx)

            image = sample["image"].unsqueeze(0).to(device)
            fg_target = (sample["target_segmentation"][fg_class_idx] > 0.5).cpu().numpy()

            out = model(image)
            seg_probs = torch.sigmoid(out["seg_logits"])[0, fg_class_idx]
            fg_ncs = 1.0 - seg_probs

            image_np = _tensor_image_to_numpy(sample["image"])

            if bool(cfg.save_original_image):
                original_path = save_dir / f"image_{sample_idx:04d}_original.png"
                _save_rgb_image(image_np, original_path)

            if bool(cfg.save_gt_mask):
                gt_mask_path = save_dir / f"image_{sample_idx:04d}_gt.png"
                _save_binary_mask(fg_target, gt_mask_path)

            alpha_masks: List[Tuple[float, np.ndarray, int, float]] = []
            reference_pixels = None
            for alpha in alpha_values:
                curve_idx = _alpha_to_curve_index(alpha, n_curve_points)
                thr_map = fg_nc_curves[curve_idx]
                conf_mask = (fg_ncs <= thr_map).detach().cpu().numpy().astype(bool)
                mask_pixels = int(conf_mask.sum())

                if reference_pixels is None:
                    reference_pixels = max(mask_pixels, 1)
                area_ratio = float(mask_pixels / reference_pixels)

                alpha_masks.append((alpha, conf_mask, mask_pixels, area_ratio))

                if bool(cfg.save_raw_masks):
                    alpha_token = str(alpha).replace(".", "p")
                    mask_path = save_dir / f"image_{sample_idx:04d}_alpha_{alpha_token}.png"
                    _save_binary_mask(conf_mask, mask_path)

                summary_rows.append(
                    {
                        "requested_index": sample_idx,
                        "source_index": source_index,
                        "alpha": alpha,
                        "curve_index": curve_idx,
                        "mask_pixels": mask_pixels,
                        "area_ratio": area_ratio,
                    }
                )

            figure_path = save_dir / f"image_{sample_idx:04d}_shrinkage.png"
            _save_image_grid(
                out_path=figure_path,
                image_np=image_np,
                gt_mask=fg_target,
                alpha_masks=alpha_masks,
                overlay_alpha=float(cfg.overlay_alpha),
                source_index=source_index,
            )
            figure_paths.append(str(figure_path))
            log.info(f"Saved shrinkage figure to: {figure_path}")

    summary_path = save_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requested_index",
                "source_index",
                "alpha",
                "curve_index",
                "mask_pixels",
                "area_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log.info(f"Wrote mask summary to: {summary_path}")

    metrics = {
        "num_images": len(image_indices),
        "num_alpha_values": len(alpha_values),
        "num_summary_rows": len(summary_rows),
    }
    objects = {
        "checkpoint_path": str(ckpt_path),
        "companion_cfg_path": str(companion_cfg_path),
        "save_dir": str(save_dir),
        "figure_paths": figure_paths,
        "summary_path": str(summary_path),
    }
    return metrics, objects


@hydra.main(version_base="1.3", config_path="../../configs", config_name="visualize.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    visualize(cfg)


if __name__ == "__main__":
    main()
