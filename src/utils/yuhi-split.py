import json
import os
import random
from pathlib import Path

# Filtered trainval split settings from README.md
n_train = 1000
n_val = 1000
area_ratio_min = 0.10
area_ratio_max = 0.60

trainval_root = Path(
    os.environ.get(
        "DATA_TRAINVAL_ROOT",
        "/home/chiba/research/kandinsky/kandinsky-data/coco/trainval",
    )
)
out_dir = trainval_root / "splits" / "yuhi"
out_dir.mkdir(exist_ok=True, parents=True)

print("loading manifest...")
with open(trainval_root / "labels.json", encoding="utf-8") as fl:
    labels = json.load(fl)

print(f"# images: {len(labels['images'])}")
print(f"# annotations: {len(labels['annotations'])}")

person_cat_id = None
for category in labels["categories"]:
    if category["name"] == "person":
        person_cat_id = category["id"]
        break

if person_cat_id is None:
    raise ValueError("could not find person category in labels.json")

person_annotations = {}
for annotation in labels["annotations"]:
    if annotation["category_id"] == person_cat_id:
        person_annotations.setdefault(annotation["image_id"], []).append(annotation)

candidate_inds = []
for i, image in enumerate(labels["images"]):
    image_annotations = person_annotations.get(image["id"], [])

    if len(image_annotations) != 1:
        continue

    image_area = image["width"] * image["height"]
    if image_area <= 0:
        continue

    person_area = sum(annotation["area"] for annotation in image_annotations)
    area_ratio = person_area / image_area
    if area_ratio_min <= area_ratio <= area_ratio_max:
        candidate_inds.append(i)

print(f"# filtered images: {len(candidate_inds)}")

random.seed(42)

if len(candidate_inds) < n_train + n_val:
    raise ValueError(
        f"not enough filtered images: need at least {n_train + n_val}, got {len(candidate_inds)}"
    )

print("sampling...")
train_inds = random.sample(candidate_inds, n_train)
rem_inds = list(set(candidate_inds) - set(train_inds))
val_inds = random.sample(rem_inds, n_val)
cal_inds = list(set(rem_inds) - set(val_inds))

print("done sampling")

print("Sanity check for overlap (should be empty)")
print(set(train_inds).intersection(set(val_inds)))
print(set(train_inds).intersection(set(cal_inds)))
print(set(val_inds).intersection(set(cal_inds)))

print("gathering images and annotations...")
train_imgs = [labels["images"][i] for i in train_inds]
val_imgs = [labels["images"][i] for i in val_inds]
cal_imgs = [labels["images"][i] for i in cal_inds]

train_ids = [image["id"] for image in train_imgs]
val_ids = [image["id"] for image in val_imgs]
cal_ids = [image["id"] for image in cal_imgs]

train_anns = [ann for ann in labels["annotations"] if ann["image_id"] in train_ids]
val_anns = [ann for ann in labels["annotations"] if ann["image_id"] in val_ids]
cal_anns = [ann for ann in labels["annotations"] if ann["image_id"] in cal_ids]

print("done gathering")

labels["images"] = None
labels["annotations"] = None

labels_train = labels.copy()
labels_val = labels.copy()
labels_cal = labels.copy()

labels_train["images"] = train_imgs
labels_train["annotations"] = train_anns

labels_val["images"] = val_imgs
labels_val["annotations"] = val_anns

labels_cal["images"] = cal_imgs
labels_cal["annotations"] = cal_anns

print("writing...")

with open(out_dir / "labels_train.json", "w", encoding="utf-8") as fl:
    json.dump(labels_train, fl)

with open(out_dir / "labels_val.json", "w", encoding="utf-8") as fl:
    json.dump(labels_val, fl)

with open(out_dir / "labels_cal.json", "w", encoding="utf-8") as fl:
    json.dump(labels_cal, fl)
