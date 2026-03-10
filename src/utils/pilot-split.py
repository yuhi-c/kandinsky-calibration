"""
pilot-split.py

Creates filtered train / calibration / test splits for the pilot experiment
described in README.md.

Filtering criteria (per README):
  - exactly 1 person in the image
  - total person-mask area ratio between 1% and 40%

Split sizes:
    - train:       300  (from trainval)
    - calibration: 300  (from trainval)
    - val:          50  (from trainval, for training monitoring)
    - test:        100  (from test / COCO-val)

If N_CAL is increased above the original 100, this script preserves the original
train/val/base-calibration samples and appends only the extra calibration images.

Outputs (trainval):
  DATA_TRAINVAL_ROOT/splits/pilot/labels_train.json
  DATA_TRAINVAL_ROOT/splits/pilot/labels_val.json
  DATA_TRAINVAL_ROOT/splits/pilot/labels_cal.json

Outputs (test):
  DATA_TEST_ROOT/splits/pilot/labels_test.json
"""

import json
import os
import random
from pathlib import Path

# -----------------------------------------------------------------------
# Paths – change these if needed, or set the env vars
# -----------------------------------------------------------------------
TRAINVAL_ROOT = Path(
    os.environ.get("DATA_TRAINVAL_ROOT", "/home/yuhix/school/Lab/kandisky-data/coco/trainval")
)
TEST_ROOT = Path(
    os.environ.get("DATA_TEST_ROOT", "/home/yuhix/school/Lab/kandisky-data/coco/test")
)

# -----------------------------------------------------------------------
# Split sizes
# -----------------------------------------------------------------------
N_TRAIN = 300
N_VAL = 50
N_CAL = 300
N_TEST = 100
BASE_N_CAL = 100

# -----------------------------------------------------------------------
# Filtering hyperparameters
# -----------------------------------------------------------------------
PERSON_CATEGORY_NAME = "person"
AREA_RATIO_MIN = 0.01   # 1%
AREA_RATIO_MAX = 0.40   # 40%
RANDOM_SEED = 42

# -----------------------------------------------------------------------


def get_person_cat_id(categories: list) -> int:
    for cat in categories:
        if cat["name"] == PERSON_CATEGORY_NAME:
            return cat["id"]
    raise ValueError("'person' category not found in labels.json")


def filter_images(labels: dict, person_cat_id: int) -> list:
    """Return list of image dicts that pass the pilot filtering rules."""
    # build lookup: image_id -> list of person annotations
    person_anns: dict[int, list] = {}
    for ann in labels["annotations"]:
        if ann["category_id"] == person_cat_id:
            person_anns.setdefault(ann["image_id"], []).append(ann)

    id_to_img = {img["id"]: img for img in labels["images"]}

    accepted = []
    for img_id, anns in person_anns.items():
        # condition 1: exactly 1 person
        if len(anns) != 1:
            continue

        img = id_to_img[img_id]
        img_area = img["width"] * img["height"]
        if img_area == 0:
            continue

        # condition 2: mask area ratio between 1% and 40%
        person_area = sum(a["area"] for a in anns)
        ratio = person_area / img_area
        if not (AREA_RATIO_MIN <= ratio <= AREA_RATIO_MAX):
            continue

        accepted.append(img)

    return accepted


def build_subset(labels: dict, selected_imgs: list, person_cat_id: int) -> dict:
    """Build a self-contained labels dict for the given image subset."""
    selected_ids = {img["id"] for img in selected_imgs}
    subset_anns = [a for a in labels["annotations"] if a["image_id"] in selected_ids]

    result = {k: v for k, v in labels.items() if k not in ("images", "annotations")}
    result["images"] = list(selected_imgs)
    result["annotations"] = subset_anns
    return result


def main():
    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Trainval split  (train + val + cal)
    # ------------------------------------------------------------------
    print("Loading trainval labels.json …")
    with open(TRAINVAL_ROOT / "labels.json") as f:
        labels_tv = json.load(f)

    person_cat_id = get_person_cat_id(labels_tv["categories"])
    print(f"  person category_id = {person_cat_id}")

    accepted_tv = filter_images(labels_tv, person_cat_id)
    print(f"  images passing filter: {len(accepted_tv)} / {len(labels_tv['images'])}")

    if N_CAL < BASE_N_CAL:
        raise ValueError(f"N_CAL must be >= {BASE_N_CAL} to preserve the original split")

    base_need_tv = N_TRAIN + N_VAL + BASE_N_CAL
    if len(accepted_tv) < base_need_tv:
        raise ValueError(
            f"Not enough filtered images: need {base_need_tv}, got {len(accepted_tv)}"
        )

    # Preserve the original split construction used when BASE_N_CAL=100:
    # first sample train/val/base-calibration together, then append only the
    # extra calibration images from the remaining pool.
    base_sampled_tv = random.sample(accepted_tv, base_need_tv)
    train_imgs = base_sampled_tv[:N_TRAIN]
    val_imgs = base_sampled_tv[N_TRAIN : N_TRAIN + N_VAL]
    base_cal_imgs = base_sampled_tv[N_TRAIN + N_VAL :]

    extra_cal_needed = N_CAL - BASE_N_CAL
    if extra_cal_needed > 0:
        used_ids = {img["id"] for img in base_sampled_tv}
        remaining_tv = [img for img in accepted_tv if img["id"] not in used_ids]
        if len(remaining_tv) < extra_cal_needed:
            raise ValueError(
                f"Not enough remaining filtered images to extend calibration: "
                f"need {extra_cal_needed}, got {len(remaining_tv)}"
            )
        extra_cal_imgs = random.sample(remaining_tv, extra_cal_needed)
    else:
        extra_cal_imgs = []

    cal_imgs = base_cal_imgs + extra_cal_imgs

    out_tv = TRAINVAL_ROOT / "splits" / "pilot"
    out_tv.mkdir(parents=True, exist_ok=True)

    print("Writing trainval splits …")
    with open(out_tv / "labels_train.json", "w") as f:
        json.dump(build_subset(labels_tv, train_imgs, person_cat_id), f)
    with open(out_tv / "labels_val.json", "w") as f:
        json.dump(build_subset(labels_tv, val_imgs, person_cat_id), f)
    with open(out_tv / "labels_cal.json", "w") as f:
        json.dump(build_subset(labels_tv, cal_imgs, person_cat_id), f)

    print(f"  train: {len(train_imgs)}, val: {len(val_imgs)}, cal: {len(cal_imgs)}")
    print(f"  → {out_tv}")

    # ------------------------------------------------------------------
    # Test split
    # ------------------------------------------------------------------
    print("\nLoading test labels.json …")
    with open(TEST_ROOT / "labels.json") as f:
        labels_test = json.load(f)

    person_cat_id_test = get_person_cat_id(labels_test["categories"])
    accepted_test = filter_images(labels_test, person_cat_id_test)
    print(f"  images passing filter: {len(accepted_test)} / {len(labels_test['images'])}")

    if len(accepted_test) < N_TEST:
        raise ValueError(
            f"Not enough filtered test images: need {N_TEST}, got {len(accepted_test)}"
        )

    test_imgs = random.sample(accepted_test, N_TEST)

    out_test = TEST_ROOT / "splits" / "pilot"
    out_test.mkdir(parents=True, exist_ok=True)

    with open(out_test / "labels_test.json", "w") as f:
        json.dump(build_subset(labels_test, test_imgs, person_cat_id_test), f)

    print(f"  test: {len(test_imgs)}")
    print(f"  → {out_test}")

    print("\nDone.")


if __name__ == "__main__":
    main()
