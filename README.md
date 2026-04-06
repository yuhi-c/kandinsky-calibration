# Preliminary Experiment: Correlation Between CP Area Stabilization and Segmentation Quality

## Overview

This document describes a preliminary experiment built on top of the [kandinsky-calibration](https://github.com/NKI-AI/kandinsky-calibration) repository.

The goal of this experiment is **not** to perform a full-scale benchmark.  
Instead, the purpose is to check the following simple hypothesis:

> Images with better segmentation quality tend to reach a stable conformal prediction (CP) area earlier when the CP threshold is varied.

More concretely, we investigate whether a curve derived from CP prediction-set area is correlated with the segmentation quality of each image, measured by **IoU**.

---

## Motivation

For a given image, CP produces a prediction set whose area changes as the threshold changes.

Intuitively:

- for **easy / well-segmented images**, the prediction-set area may shrink quickly and become stable early,
- for **hard / poorly-segmented images**, the area may keep changing for longer and stabilize later.

If this tendency exists, then the **stabilization behavior of the CP area curve** may be useful as a proxy for image-level uncertainty.

This README focuses on verifying that idea with a lightweight experiment.

---

## Main Hypothesis

We test the following claim:

> For images with high IoU, the CP prediction-set area becomes stable earlier.  
> For images with low IoU, the area stabilizes later.

Equivalently:

- high IoU -> small stabilization time
- low IoU -> large stabilization time

---

## Experimental Setting

### Task

Binary segmentation on **COCO-person**.

### Why COCO-person?

This dataset is used because:

- it is easy to convert into a binary segmentation task,
- the kandinsky-calibration repository is already designed around COCO-person,
- it is suitable for a first correlation study.

### Nature of This Experiment

This is a **preliminary / pilot experiment**.

We only want to know whether there is a rough correlation trend.  
We do **not** aim to claim final results from this experiment.

---

## Definition of the Curve

For an image `x` and CP threshold `a`, let:

- `S_x(a)` = CP prediction set for image `x` at threshold `a`
- `A_x(a) = |S_x(a)|` = area (number of pixels) of the prediction set

Then the basic area curve is:

`a -> A_x(a)`

To normalize across images, define the area ratio as:

`r_x(a) = A_x(a) / (A_x(a_min) + eta)`

where:

- `a_min` is the smallest threshold in the sweep,
- `eta` is a small constant to avoid division by zero.

Thus, `r_x(a)` starts near `1` and decreases as the prediction set shrinks.

---

## Area-Change Curve

We define the local area change as:

`d_x(a) = |r_x(a + Delta) - r_x(a)|`

where `Delta` is the threshold step.

This quantity measures how much the normalized CP area changes between adjacent thresholds.

---

## Stabilization Time

Our main metric is the **stabilization time** of the area curve.

For image `x`, define:

`tau_stab(x) = min { a | d_x(a), d_x(a+Delta), ..., d_x(a+(w-1)Delta) <= epsilon, and 1 - r_x(a) >= rho }`

### Interpretation

This means:

1. the area change remains small for `w` consecutive steps  
   -> the curve has entered a stable region

2. the area has already shrunk by at least `rho`  
   -> we exclude curves that are flat only because they never meaningfully changed

### Parameters

- `w`: window size for consecutive stability
- `epsilon`: tolerance for small area change
- `rho`: minimum shrinkage ratio before stability can be declared

These hyperparameters will be fixed later for the pilot experiment.

---

## Segmentation Quality

For each test image, we compute:

- the **initial segmentation IoU**
- the curve-based metric `tau_stab(x)`

Then we study whether these two values are correlated.

---

## What Will Be Evaluated

For each test image, we will compute:

1. **IoU** of the initial segmentation
2. **stabilization time** `tau_stab(x)` from the CP area curve

Then we will analyze the following:

### 1. Scatter Plot

- x-axis: IoU
- y-axis: `tau_stab(x)`

Expected trend:

- high-IoU images should appear lower on the plot
- low-IoU images should appear higher on the plot


### 2. Optional Box Plot (Optional, not implement now)

We may additionally split images into:

- high-IoU group
- low-IoU group

and compare the distribution of `tau_stab`.

---


### Initial Split

- **train**: 1000 images
- **calibration**: 1000 images
- **test**: 2000 images

These follow the original paper's setting

---

## COCO-person Filtering Rule

To avoid extremely difficult or degenerate examples in the first pilot run, and considering this is just experiment, we filter candidate images using the following conditions:

- the image contains at least **one person**
- total person-mask area ratio is between **10% and 60%**
- the number of persons is **1**

After filtering, we randomly sample images and split them into:

- train: 1000
- validation: 1000
- calibration: all remaining filtered trainval images


## Training Strategy

1. train a segmentation model on a small COCO-person subset,
2. Implement pixcel-wise calibration
3. evaluate the CP behavior on the test set.

We want a model that produces a reasonable spread of IoU values, so that the correlation with the curve-derived metric can be observed.

---

## Planned Workflow

### Step 1. Prepare a Small COCO-person Subset

Create filtered train / calibration / test splits satisfying the conditions above.

### Step 2. Train a Base Segmentation Model

Train a base segmentation model on the small COCO-person training subset using the kandinsky-calibration repository.

### Step 3. Run Pixel-wise Calibration

Run **pixel-wise calibration** on the calibration split.

In this experiment, we do **not** use the full Kandinsky grouping procedure.  
Instead, we only use the **pixel-wise calibration** stage and obtain a non-conformity curve for each pixel location.

More specifically, calibration produces `nc_curves`, which store pixel-wise threshold values across multiple confidence / quantile levels.

**Minimal command (after training)**

This writes a calibrated checkpoint `cmodel.ckpt` under `logs/calibrate/runs/.../`.

```bash
python src/calibrate.py ckpt_path=/path/to/trained_model.ckpt
```

### Step 4. Sweep the Calibration Levels and Build Prediction Sets

For each test image:

- sweep the calibration level `a`,
- retrieve the corresponding pixel-wise threshold map from `nc_curves`,
- construct the prediction set `S_x(a)` by checking, at each pixel, whether the model output is large enough relative to the threshold,
- compute the area `A_x(a) = |S_x(a)|`,
- compute the normalized area ratio `r_x(a)`,
- compute the area-change measure `d_x(a)`,
- compute `tau_stab(x)`.

More concretely, if the model output probability at a pixel is `p` and the calibrated non-conformity threshold at level `a` is `q_a`, then the pixel is included in the mask when

`1 - p <= q_a`

which is equivalently written as

`p >= 1 - q_a`.

**Minimal command (generate per-image area curves)**

This produces per-image PNGs and a `summary.csv` (IoU + `tau_stab`) under `logs/area_curve/runs/.../area_curves/`.

```bash
python src/area_curve.py ckpt_path=/path/to/logs/calibrate/runs/.../cmodel.ckpt
```

Useful overrides:

```bash
# process only first 100 test images
python src/area_curve.py ckpt_path=... max_images=100

# change alpha sweep and stabilization hyperparameters
python src/area_curve.py ckpt_path=... alpha_max=0.8 alpha_steps=81 w=5 epsilon=0.005 rho=0.2
```

### Step 5. Compare Against IoU

For each test image:

- collect IoU
- collect `tau_stab(x)`

Then generate:

- scatter plot
- Pearson correlation
- Spearman correlation
- optional box plot

---

## Quick Commands

Below is the actual command sequence for this pilot experiment.

### 0. Load environment variables

```bash
source ~/.zshrc
conda activate kandinsky
```

### 1. Create the pilot split

```bash
python src/utils/pilot-split.py
```

This creates:

- `train`: 1000 images
- `val`: 1000 images
- `calibration`: All other remaining images

### 2. Train the pilot model

```bash
python src/train.py experiment=train_pilot
```

Checkpoint output:

```bash
logs/train/runs/train_pilot/<timestamp>/checkpoints/last.ckpt
```

### 3. Evaluate the trained model

```bash
python src/eval.py experiment=eval_pilot ckpt_path=logs/train/runs/train_pilot/<timestamp>/checkpoints/last.ckpt
```

Output directory:

```bash
logs/eval/runs/eval_pilot/<timestamp>/
```

### 4. Run pixel-wise calibration

```bash
python src/calibrate.py experiment=cal_pilot ckpt_path=logs/train/runs/train_pilot/<timestamp>/checkpoints/last.ckpt
```

Calibrated checkpoint:

```bash
logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt
```

### 5. Generate area curves

Important: `area_curve.py` needs `+experiment=...` instead of `experiment=...`.

```bash
python src/area_curve.py +experiment=area_curve_pilot ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt
```

Current recommended sweep / stabilization settings:

```bash
python src/area_curve.py +experiment=area_curve_pilot \
   ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt \
   alpha_max=0.9 alpha_steps=51 w=3 epsilon=0.005 rho=0.1
```

Output directory:

```bash
logs/area_curve/runs/area_curve_pilot/<timestamp>/area_curves/
```

Important outputs:

```bash
logs/area_curve/runs/area_curve_pilot/<timestamp>/area_curves/summary.csv
logs/area_curve/runs/area_curve_pilot/<timestamp>/area_curves/curves/*.png
```

### 6. Plot scatter from summary.csv

Default scatter:

```bash
python src/utils/plot_summary_scatter.py \
   --summary_csv logs/area_curve/runs/area_curve_pilot/<timestamp>/area_curves/summary.csv
```

Plot any metric against IoU:

```bash
python src/utils/plot_summary_scatter.py \
   --summary_csv logs/area_curve/runs/area_curve_pilot/<timestamp>/area_curves/summary.csv \
   --x_col iou \
   --y_col auc_area_ratio
```

Other useful `y_col` values:

- `tau_stab`
- `auc_area_ratio`
- `alpha_at_10_shrink`
- `alpha_at_20_shrink`
- `alpha_at_50_shrink`

### 7. Learn an IoU classification rule from calibration data

This step fits a rule of the form:

```bash
area_ratio(alpha=a) >= threshold  =>  IoU <= x
```

The rule is selected on the calibration split, then applied to the test split.

```bash
python src/iou_rule.py experiment=iou_rule_pilot \
   ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt \
   iou_threshold=0.3
```

In the current pilot setup, the rule search is restricted to:

```bash
alpha in [0.05, 0.3]
```

Useful overrides:

```bash
# try a different IoU cutoff
python src/iou_rule.py experiment=iou_rule_pilot \
   ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt \
   iou_threshold=0.2

# optimize a different metric
python src/iou_rule.py experiment=iou_rule_pilot \
   ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt \
   iou_threshold=0.3 \
   selection_metric=f1

# allow the search to decide whether large or small area_ratio indicates low IoU
python src/iou_rule.py experiment=iou_rule_pilot \
   ckpt_path=logs/calibrate/runs/cal_pilot/<timestamp>/cmodel.ckpt \
   iou_threshold=0.3 \
   direction=auto
```

IoU-rule outputs:

```bash
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/selected_rule.json
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/rule_search.csv
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/calibration_predictions.csv
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/test_predictions.csv
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/classification_report.txt
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/calibration_rule_scatter.png
logs/iou_rule/runs/iou_rule_pilot/<timestamp>/iou_rule/test_rule_scatter.png
```

---

## Expected Result

The expected trend is:

- **high IoU images** stabilize earlier
- **low IoU images** stabilize later

That is:

`IoU increases -> tau_stab decreases`  
`IoU decreases -> tau_stab increases`

If this tendency is visible even roughly, then the area-stabilization behavior may serve as a useful uncertainty-related image-level signal.

---

## Items Still To Be Decided

The following parts are not fixed yet.

### 1. Threshold Sweep Design

We still need to decide how to choose the threshold values `a`.

Possible options:

- uniformly spaced thresholds
- quantile-based thresholds

### 2. Hyperparameters of `tau_stab`

The following parameters must be determined:

- `w`
- `epsilon`
- `rho`

Since this is only a pilot study, these values do not need to be perfectly optimized.  
We only need values that are reasonable enough to check whether the trend exists.

---