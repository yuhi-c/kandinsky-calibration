from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, ListConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.curves.area_curve import (
    _alpha_to_curve_index,
    _build_eval_loader,
    _build_tau_stab_specs,
    _early_slope,
    _knee_by_second_diff,
    _plot_curve,
    _safe_iou,
    _write_curve_csv,
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


def _parse_alpha_steps_values(alpha_steps_cfg: Any) -> List[int]:
    if isinstance(alpha_steps_cfg, ListConfig):
        raw_values = list(alpha_steps_cfg)
    elif isinstance(alpha_steps_cfg, (list, tuple)):
        raw_values = list(alpha_steps_cfg)
    else:
        raw_values = [alpha_steps_cfg]

    alpha_steps_values: List[int] = []
    for raw in raw_values:
        value = int(raw)
        if value < 2:
            raise ValueError("Each alpha_steps value must be >= 2")
        alpha_steps_values.append(value)

    if not alpha_steps_values:
        raise ValueError("alpha_steps cannot be empty")

    return alpha_steps_values


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

    fg_class_idx = int(cfg.fg_class_idx)
    nc_curves: torch.Tensor = ckpt["nc_curves"]
    n_curve_points = int(nc_curves.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)
    # Only the requested foreground class is used downstream, so keep
    # background curves off GPU to reduce VRAM pressure for large checkpoints.
    fg_nc_curves = nc_curves[:, fg_class_idx].to(device)

    alpha_min = float(cfg.alpha_min)
    alpha_max = float(cfg.alpha_max)
    if not (0.0 <= alpha_min <= alpha_max <= 1.0):
        raise ValueError("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1")
    alpha_steps_values = _parse_alpha_steps_values(cfg.alpha_steps)
    total_alpha_tasks = len(alpha_steps_values)
    log.info(
        f"Planned alpha-step tasks: total={total_alpha_tasks}, values={alpha_steps_values}"
    )

    eta = float(cfg.eta)
    tau_stab_specs = _build_tau_stab_specs(cfg)
    extra_metric_names = [
        spec.metric_name for spec in tau_stab_specs if spec.metric_name != "tau_stab"
    ]
    tau_patterns_per_alpha_task = len(tau_stab_specs)
    total_pattern_tasks = total_alpha_tasks * tau_patterns_per_alpha_task
    log.info(
        "Planned sweep pattern combinations: "
        f"per_alpha_steps={tau_patterns_per_alpha_task}, total={total_pattern_tasks}"
    )

    base_out_dir = Path(cfg.paths.output_dir) / "area_curv_stab"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    save_curve_png = bool(cfg.get("save_curve_png", False))
    save_curve_csv = bool(cfg.get("save_curve_csv", False))

    max_images = cfg.get("max_images")
    max_images = int(max_images) if max_images is not None else None
    progress_log_interval = int(cfg.get("progress_log_interval", 50))
    if progress_log_interval <= 0:
        progress_log_interval = 1

    dataset_size: int | None = None
    if hasattr(eval_loader, "dataset"):
        try:
            dataset_size = len(eval_loader.dataset)
        except TypeError:
            dataset_size = None

    if dataset_size is not None:
        total_images = min(dataset_size, max_images) if max_images is not None else dataset_size
    else:
        total_images = max_images

    if total_images is not None:
        log.info(f"Planned images per task: total={total_images}")
    else:
        log.info(
            "Planned images per task: total=unknown "
            "(dataset length unavailable and max_images unset)"
        )

    task_metric_dicts: List[Dict[str, Any]] = []
    task_summary_paths: List[Path] = []

    for task_idx, alpha_steps in enumerate(alpha_steps_values, start=1):
        alpha_done_before = task_idx - 1
        alpha_remaining_before = total_alpha_tasks - alpha_done_before
        pattern_done_before = alpha_done_before * tau_patterns_per_alpha_task
        pattern_remaining_before = total_pattern_tasks - pattern_done_before
        log.info(
            f"Task progress: alpha_steps_done={alpha_done_before}/{total_alpha_tasks}, "
            f"alpha_steps_remaining={alpha_remaining_before}, "
            f"pattern_done={pattern_done_before}/{total_pattern_tasks}, "
            f"pattern_remaining={pattern_remaining_before}, "
            f"starting alpha_steps={alpha_steps}"
        )

        alphas = np.linspace(alpha_min, alpha_max, alpha_steps)
        if total_alpha_tasks > 1:
            out_dir = base_out_dir / f"task_{task_idx:03d}_alpha_steps_{alpha_steps}"
        else:
            out_dir = base_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = out_dir / "curves"
        if save_curve_png:
            curves_dir.mkdir(parents=True, exist_ok=True)
        curve_csv_dir = out_dir / "curve_csvs"
        if save_curve_csv:
            curve_csv_dir.mkdir(parents=True, exist_ok=True)

        results: List[ImageTauStabResult] = []
        image_counter = 0

        def _log_progress(done: int, *, force: bool = False) -> None:
            if done <= 0:
                return
            if not force and done % progress_log_interval != 0:
                return

            if total_images is None:
                log.info(
                    f"[alpha_steps={alpha_steps}] Progress: done={done}, remaining=unknown"
                )
                return

            remaining = max(total_images - done, 0)
            log.info(
                f"[alpha_steps={alpha_steps}] Progress: "
                f"done={done}/{total_images}, remaining={remaining}"
            )

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
                    curve_indices = np.zeros_like(alphas, dtype=np.int64)
                    for i, alpha in enumerate(alphas):
                        idx = _alpha_to_curve_index(alpha, n_curve_points)
                        curve_indices[i] = idx
                        thr_map = fg_nc_curves[idx]
                        conf_mask = fg_ncs[b] <= thr_map
                        areas[i] = float(conf_mask.sum().item())

                    area0 = float(areas[0])
                    area_ratios = areas / (area0 + eta)
                    slope_0_3 = _early_slope(alphas, area_ratios, end_idx=3)
                    knee_alpha_second_diff, _ = _knee_by_second_diff(alphas, area_ratios)

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

                    if save_curve_png:
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

                    if save_curve_csv:
                        curve_csv_path = curve_csv_dir / f"image_{image_counter:05d}.csv"
                        _write_curve_csv(
                            out_path=curve_csv_path,
                            image_idx=image_counter,
                            alphas=alphas,
                            areas=areas,
                            area_ratios=area_ratios,
                            iou=iou,
                            gt_pixels=gt_pixels,
                            tau_stab=tau,
                            curve_indices=curve_indices,
                        )

                    results.append(
                        ImageTauStabResult(
                            image_idx=image_counter,
                            iou=iou,
                            gt_pixels=gt_pixels,
                            tau_stab=tau,
                            extra_metrics={k: v for k, v in tau_metrics.items() if k != "tau_stab"},
                        )
                    )
                    image_counter += 1
                    _log_progress(image_counter)

                if max_images is not None and image_counter >= max_images:
                    break

        _log_progress(image_counter, force=True)
        if total_images is not None:
            log.info(
                f"[alpha_steps={alpha_steps}] Finished processing: "
                f"done={image_counter}/{total_images}, "
                f"remaining={max(total_images - image_counter, 0)}"
            )
        else:
            log.info(f"[alpha_steps={alpha_steps}] Finished processing: done={image_counter}, remaining=unknown")

        summary_path = out_dir / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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
            "alpha_steps": alpha_steps,
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

        task_metric_dicts.append(metric_dict)
        task_summary_paths.append(summary_path)

        alpha_done_after = task_idx
        alpha_remaining_after = total_alpha_tasks - alpha_done_after
        pattern_done_after = alpha_done_after * tau_patterns_per_alpha_task
        pattern_remaining_after = total_pattern_tasks - pattern_done_after
        log.info(
            f"Task progress: alpha_steps_done={alpha_done_after}/{total_alpha_tasks}, "
            f"alpha_steps_remaining={alpha_remaining_after}, "
            f"pattern_done={pattern_done_after}/{total_pattern_tasks}, "
            f"pattern_remaining={pattern_remaining_after}, "
            f"finished alpha_steps={alpha_steps}"
        )
        log.info(f"Wrote tau_stab summary to: {summary_path}")
        if save_curve_png:
            log.info(f"Wrote per-image curve PNGs to: {curves_dir}")
        if save_curve_csv:
            log.info(f"Wrote per-image curve CSVs to: {curve_csv_dir}")

    if total_alpha_tasks > 1:
        sweep_summary_path = base_out_dir / "summary_by_alpha_steps.csv"
        sweep_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with sweep_summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_index",
                    "alpha_steps",
                    "num_images",
                    "mean_iou",
                    "mean_gt_pixels",
                    "mean_tau_stab",
                    *[f"mean_{metric_name}" for metric_name in extra_metric_names],
                    "summary_path",
                ],
            )
            writer.writeheader()
            for task_index, (task_metric, summary_path) in enumerate(
                zip(task_metric_dicts, task_summary_paths),
                start=1,
            ):
                row = {
                    "task_index": task_index,
                    "alpha_steps": task_metric["alpha_steps"],
                    "num_images": task_metric["num_images"],
                    "mean_iou": task_metric["mean_iou"],
                    "mean_gt_pixels": task_metric["mean_gt_pixels"],
                    "mean_tau_stab": task_metric["mean_tau_stab"],
                    "summary_path": str(summary_path),
                }
                for metric_name in extra_metric_names:
                    row[f"mean_{metric_name}"] = task_metric[f"mean_{metric_name}"]
                writer.writerow(row)

        metric_dict = {
            "num_alpha_step_tasks": total_alpha_tasks,
            "num_pattern_tasks": total_pattern_tasks,
            "mean_iou_over_tasks": float(
                np.nanmean([task_metric["mean_iou"] for task_metric in task_metric_dicts])
            ),
            "mean_tau_stab_over_tasks": float(
                np.nanmean([task_metric["mean_tau_stab"] for task_metric in task_metric_dicts])
            ),
        }
        object_dict = {
            "cfg": cfg,
            "eval_source": eval_source_name,
            "out_dir": str(base_out_dir),
            "summary_path": str(sweep_summary_path),
            "task_summary_paths": [str(path) for path in task_summary_paths],
        }
        log.info(f"Wrote alpha_steps sweep summary to: {sweep_summary_path}")
        return metric_dict, object_dict

    object_dict = {
        "cfg": cfg,
        "eval_source": eval_source_name,
        "out_dir": str(base_out_dir),
        "summary_path": str(task_summary_paths[0]),
    }
    return task_metric_dicts[0], object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="curv/area_curv_stab.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_area_curv_stab(cfg)


if __name__ == "__main__":
    main()
