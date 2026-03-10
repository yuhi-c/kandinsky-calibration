from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@dataclass(frozen=True)
class SplitCurves:
    image_indices: np.ndarray
    ious: np.ndarray
    area_ratios: np.ndarray
    alphas: np.ndarray


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


def _compute_split_curves(
    *,
    loader: DataLoader,
    model,
    nc_curves: torch.Tensor,
    alphas: np.ndarray,
    fg_class_idx: int,
    base_seg_threshold: float,
    eta: float,
    device: torch.device,
    max_images: int | None,
) -> SplitCurves:
    image_indices: List[int] = []
    ious: List[float] = []
    area_ratios_all: List[np.ndarray] = []

    fg_nc_curves = nc_curves[:, fg_class_idx]
    n_curve_points = int(nc_curves.shape[0])
    image_counter = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target_segmentation"].to(device)

            out = model(images)
            seg_probs = torch.sigmoid(out["seg_logits"])

            fg_probs = seg_probs[:, fg_class_idx]
            fg_targets = targets[:, fg_class_idx] > 0.5
            fg_ncs = 1.0 - fg_probs

            for b in range(images.shape[0]):
                if max_images is not None and image_counter >= max_images:
                    break

                base_pred_mask = fg_probs[b] >= base_seg_threshold
                iou = _safe_iou(base_pred_mask, fg_targets[b])

                areas = np.zeros_like(alphas, dtype=np.float64)
                for i, alpha in enumerate(alphas):
                    idx = _alpha_to_curve_index(alpha, n_curve_points)
                    thr_map = fg_nc_curves[idx]
                    conf_mask = fg_ncs[b] <= thr_map
                    areas[i] = float(conf_mask.sum().item())

                area0 = float(areas[0])
                area_ratios = areas / (area0 + eta)

                image_indices.append(image_counter)
                ious.append(iou)
                area_ratios_all.append(area_ratios)
                image_counter += 1

            if max_images is not None and image_counter >= max_images:
                break

    return SplitCurves(
        image_indices=np.asarray(image_indices, dtype=int),
        ious=np.asarray(ious, dtype=float),
        area_ratios=np.asarray(area_ratios_all, dtype=float),
        alphas=np.asarray(alphas, dtype=float),
    )


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.logical_and(y_true, y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")

    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    if np.isfinite(recall) and np.isfinite(specificity):
        balanced_accuracy = 0.5 * (recall + specificity)
    else:
        balanced_accuracy = float("nan")

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _select_metric(metrics: Dict[str, float], metric_name: str) -> float:
    value = metrics.get(metric_name, float("nan"))
    return float(value) if np.isfinite(value) else float("-inf")


def _search_rule(
    *,
    split_curves: SplitCurves,
    iou_threshold: float,
    n_score_thresholds: int,
    selection_metric: str,
    direction: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    y_true = split_curves.ious <= iou_threshold
    search_rows: List[Dict[str, Any]] = []
    best_rule: Dict[str, Any] | None = None
    best_value = float("-inf")

    if direction not in {"ge", "le", "auto"}:
        raise ValueError("direction must be one of: ge, le, auto")

    directions = [direction] if direction != "auto" else ["ge", "le"]

    for alpha_idx, alpha in enumerate(split_curves.alphas):
        scores = split_curves.area_ratios[:, alpha_idx]
        quantiles = np.linspace(0.0, 1.0, int(n_score_thresholds))
        thresholds = np.unique(np.quantile(scores, quantiles))

        for thr in thresholds:
            for d in directions:
                if d == "ge":
                    y_pred = scores >= thr
                else:
                    y_pred = scores <= thr

                metrics = _compute_binary_metrics(y_true, y_pred)
                row = {
                    "alpha": float(alpha),
                    "score_threshold": float(thr),
                    "direction": d,
                    "iou_threshold": float(iou_threshold),
                    **metrics,
                }
                search_rows.append(row)

                value = _select_metric(metrics, selection_metric)
                if value > best_value:
                    best_value = value
                    best_rule = row

    if best_rule is None:
        raise RuntimeError("Failed to find a valid rule on calibration split")

    return best_rule, search_rows


def _apply_rule(
    *,
    split_curves: SplitCurves,
    rule: Dict[str, Any],
    iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    alpha_idx = int(np.argmin(np.abs(split_curves.alphas - float(rule["alpha"]))))
    scores = split_curves.area_ratios[:, alpha_idx]
    if rule["direction"] == "ge":
        y_pred = scores >= float(rule["score_threshold"])
    else:
        y_pred = scores <= float(rule["score_threshold"])

    y_true = split_curves.ious <= iou_threshold
    metrics = _compute_binary_metrics(y_true, y_pred)

    rows: List[Dict[str, Any]] = []
    for image_idx, iou, score, pred, actual in zip(
        split_curves.image_indices,
        split_curves.ious,
        scores,
        y_pred,
        y_true,
    ):
        rows.append(
            {
                "image_idx": int(image_idx),
                "iou": float(iou),
                "score": float(score),
                "pred_low_iou": int(bool(pred)),
                "actual_low_iou": int(bool(actual)),
            }
        )

    return rows, metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_split_report(
    *,
    split_name: str,
    rows: List[Dict[str, Any]],
    metrics: Dict[str, float],
    rule: Dict[str, Any],
    iou_threshold: float,
) -> str:
    lines = [
        f"[{split_name}]",
        f"rule: area_ratio(alpha={float(rule['alpha']):.3f}) {rule['direction']} {float(rule['score_threshold']):.6f} => IoU <= {iou_threshold:.3f}",
        (
            "metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"specificity={metrics['specificity']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        ),
        f"confusion_matrix: tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}",
        "sample_predictions:",
    ]

    for row in rows[:15]:
        lines.append(
            "  "
            f"image_idx={row['image_idx']}, iou={row['iou']:.4f}, score={row['score']:.6f}, "
            f"pred_low_iou={row['pred_low_iou']}, actual_low_iou={row['actual_low_iou']}"
        )

    return "\n".join(lines)


def _plot_rule_scatter(path: Path, rows: List[Dict[str, Any]], iou_threshold: float, rule: Dict[str, Any], title: str) -> None:
    ious = np.asarray([row["iou"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_low_iou"] for row in rows], dtype=int)

    plt.figure(figsize=(6, 5))
    plt.scatter(ious[actual == 0], scores[actual == 0], alpha=0.7, s=18, label=f"IoU > {iou_threshold}")
    plt.scatter(ious[actual == 1], scores[actual == 1], alpha=0.7, s=18, label=f"IoU <= {iou_threshold}")
    plt.axvline(iou_threshold, linestyle="--", color="gray")
    plt.axhline(float(rule["score_threshold"]), linestyle="--", color="black")
    plt.xlabel("IoU")
    plt.ylabel(f"area_ratio(alpha={float(rule['alpha']):.3f})")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


@utils.task_wrapper
def run_iou_rule(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Loading checkpoint: {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    if "nc_curves" not in ckpt:
        raise KeyError("Checkpoint does not contain 'nc_curves'. Run calibration first.")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    nc_curves = ckpt["nc_curves"].to(device)

    alphas = np.linspace(float(cfg.alpha_min), float(cfg.alpha_max), int(cfg.alpha_steps))
    fg_class_idx = int(cfg.fg_class_idx)
    eta = float(cfg.eta)
    base_seg_threshold = float(cfg.base_seg_threshold)
    iou_threshold = float(cfg.iou_threshold)

    datamodule.setup(stage="calibrate")
    cal_loader = datamodule.val_dataloader()
    calibration_curves = _compute_split_curves(
        loader=cal_loader,
        model=model,
        nc_curves=nc_curves,
        alphas=alphas,
        fg_class_idx=fg_class_idx,
        base_seg_threshold=base_seg_threshold,
        eta=eta,
        device=device,
        max_images=cfg.get("max_cal_images"),
    )

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    test_curves = _compute_split_curves(
        loader=test_loader,
        model=model,
        nc_curves=nc_curves,
        alphas=alphas,
        fg_class_idx=fg_class_idx,
        base_seg_threshold=base_seg_threshold,
        eta=eta,
        device=device,
        max_images=cfg.get("max_test_images"),
    )

    best_rule, search_rows = _search_rule(
        split_curves=calibration_curves,
        iou_threshold=iou_threshold,
        n_score_thresholds=int(cfg.n_score_thresholds),
        selection_metric=str(cfg.selection_metric),
        direction=str(cfg.direction),
    )

    cal_rows, cal_metrics = _apply_rule(
        split_curves=calibration_curves,
        rule=best_rule,
        iou_threshold=iou_threshold,
    )
    test_rows, test_metrics = _apply_rule(
        split_curves=test_curves,
        rule=best_rule,
        iou_threshold=iou_threshold,
    )

    out_dir = Path(cfg.paths.output_dir) / "iou_rule"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        out_dir / "rule_search.csv",
        search_rows,
        [
            "alpha",
            "score_threshold",
            "direction",
            "iou_threshold",
            "tp",
            "tn",
            "fp",
            "fn",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "f1",
            "balanced_accuracy",
        ],
    )
    _write_csv(
        out_dir / "calibration_predictions.csv",
        cal_rows,
        ["image_idx", "iou", "score", "pred_low_iou", "actual_low_iou"],
    )
    _write_csv(
        out_dir / "test_predictions.csv",
        test_rows,
        ["image_idx", "iou", "score", "pred_low_iou", "actual_low_iou"],
    )

    with (out_dir / "selected_rule.json").open("w") as f:
        json.dump(
            {
                "selected_rule": best_rule,
                "selection_metric": str(cfg.selection_metric),
                "calibration_metrics": cal_metrics,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )

    _plot_rule_scatter(
        out_dir / "calibration_rule_scatter.png",
        cal_rows,
        iou_threshold,
        best_rule,
        title="Calibration split",
    )
    _plot_rule_scatter(
        out_dir / "test_rule_scatter.png",
        test_rows,
        iou_threshold,
        best_rule,
        title="Test split",
    )

    calibration_report = _build_split_report(
        split_name="calibration",
        rows=cal_rows,
        metrics=cal_metrics,
        rule=best_rule,
        iou_threshold=iou_threshold,
    )
    test_report = _build_split_report(
        split_name="test",
        rows=test_rows,
        metrics=test_metrics,
        rule=best_rule,
        iou_threshold=iou_threshold,
    )

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(calibration_report + "\n\n" + test_report + "\n")

    log.info("Selected IoU rule:")
    log.info(
        "area_ratio(alpha=%.3f) %s %.6f => IoU <= %.3f",
        float(best_rule["alpha"]),
        best_rule["direction"],
        float(best_rule["score_threshold"]),
        iou_threshold,
    )
    log.info("Calibration metrics: %s", cal_metrics)
    log.info("Test metrics: %s", test_metrics)
    log.info("Test classification report:\n%s", test_report)
    log.info("Wrote classification report to: %s", report_path)

    metric_dict = {
        "selected_alpha": float(best_rule["alpha"]),
        "selected_score_threshold": float(best_rule["score_threshold"]),
        "calibration_balanced_accuracy": float(cal_metrics["balanced_accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
    }
    object_dict = {
        "cfg": cfg,
        "out_dir": str(out_dir),
        "classification_report": str(report_path),
    }
    log.info(f"Wrote IoU rule outputs to: {out_dir}")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="iou_rule.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    run_iou_rule(cfg)


if __name__ == "__main__":
    main()