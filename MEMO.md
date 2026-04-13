original train method
python src/train.py experiment=train_coco-person_t1000

t1000 is in research/kandinsky/kandinsky-data/coco/trainval

- calibration
for calibration there is for config


## Flow of experiment
1. Coco-2017 has train and validation datas first of all
2. coco-prepare.py loads them into this repo
3. coco-split.py split them into  train(1000), val(1000), and cal(rest of them), place it outside the repo
4. 
```
python src/train.py experiment=train_coco-person_yuhi
```
If you get error, use this command before training
```
export DATA_TRAINVAL_ROOT=/home/chiba/research/kandinsky/kandinsky-data/coco/trainval
export DATA_TEST_ROOT=/home/chiba/research/kandinsky/kandinsky-data/coco/test
```
train model with 1000 train data

5. For calibration
```
CUDA_VISIBLE_DEVICES=1 python src/calibrate.py experiment=cal_coco-person_t1000_c2000_pixel_yuhi ckpt_path=logs/train/runs/train_coco-person_yuhi/2026-04-05_01-08-20/checkpoints/last.ckpt
```
Because of our experiment, we only use pixel wise calibraiton

For more steps
```
CUDA_VISIBLE_DEVICES=1 python3 src/calibrate.py \
  ckpt_path=logs/train/runs/train_coco-person_t1000/epok269-1259/checkpoints/epoch_749.ckpt
  experiment=cal_coco-person_t1000_c2000_pixel \
  callbacks.calibrate.nc_curve_points=30000
```
/home/chiba/research/kandinsky/kandinsky-calibration/logs/train/runs/train_coco-person_t1000/epok269-1259/checkpoints/epoch_749.ckpt
6. Curv creating
```
CUDA_VISIBLE_DEVICES=1 python src/area_curve.py ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel_yuhi/2026-04-05_15-55-10/cmodel.ckpt
```

or
```
CUDA_VISIBLE_DEVICES=1 python src/area_curve.py \
  ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel_yuhi/2026-04-06_13-17-38/cmodel.ckpt \
  experiment=area_curve_coco-person_yuhi \
  'tau_stab_sweep.enabled=true' \
  'tau_stab_sweep.w_values=[1,2,3,4,5]' \
  'tau_stab_sweep.epsilon_values=[0.001,0.005,0.01]' \
  'tau_stab_sweep.rho_values=[0.05,0.1,0.2]'

```

- Stub Only
```
CUDA_VISIBLE_DEVICES=0  python src/area_curv_stab.py \
  ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel/step20k/cmodel.ckpt \
  experiment=area_curv_stab_coco-person_yuhi \
  'tau_stab_sweep.enabled=true' \
  'alpha_steps=[6000,7000,8000,9000,10000,12000,14000]' \
  data.eval_source=trainval \
  data.eval_subset=cal \
  +data.eval_sample_size=100 \
  +data.eval_sample_seed=3 \
  max_images=100 \
  'tau_stab_sweep.w_values=[2,3,5,7,9,11,13,15]' \
  'tau_stab_sweep.epsilon_values=[0.000001,0.000005,0.00001,0.00003,0.00005,0.00007,0.00009,0.0001,0.0005,0.001,0.0015,0.002]' \
  'tau_stab_sweep.rho_values=[0.05,0.1,0.15,0.2]' \
  +progress_log_interval=50
```

```
CUDA_VISIBLE_DEVICES=1  python src/area_curv_stab.py \
  ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel/step20k/cmodel.ckpt \
  experiment=area_curv_stab_coco-person_yuhi \
  'tau_stab_sweep.enabled=true' \
  'alpha_steps=[15000,16000,17000,18000,19000,20000]' \
  data.eval_source=trainval \
  data.eval_subset=cal \
  +data.eval_sample_size=100 \
  +data.eval_sample_seed=3 \
  max_images=100 \
  'tau_stab_sweep.w_values=[2,3,5,7,9,11,13,15]' \
  'tau_stab_sweep.epsilon_values=[0.000001,0.000005,0.00001,0.00003,0.00005,0.00007,0.00009,0.0001,0.0005,0.001,0.0015,0.002]' \
  'tau_stab_sweep.rho_values=[0.05,0.1,0.15,0.2]' \
  +progress_log_interval=50
```


- use different images for curv
dot not set seed into 42
```
python src/area_curv_stab.py \
  ckpt_path=/mnt/data2/logs/calibrate/cmodel.ckpt \
  experiment=area_curv_stab_coco-person_yuhi \
  data.eval_source=trainval \
  data.eval_subset=cal \
  +data.eval_sample_size=100 \
  +data.eval_sample_seed=3 \
  alpha_steps=9000 \
  w=9 epsilon=0.00009 rho=0.05 \
  tau_stab_sweep.enabled=false \
  +progress_log_interval=10
```

Use more image
```
python src/area_curv_stab.py \
  ckpt_path=/mnt/data2/logs/calibrate/cmodel.ckpt \
  experiment=area_curv_stab_coco-person_yuhi \
  data.eval_source=trainval \
  data.eval_subset=cal \
  +data.eval_sample_size=500 \
  +data.eval_sample_seed=3 \
  max_images=500 \
  alpha_steps=9000 \
  w=9 epsilon=0.00009 rho=0.05 \
  tau_stab_sweep.enabled=false \
  +progress_log_interval=10

```

Use whole image
```
python src/area_curv_stab.py \
  ckpt_path=/mnt/data2/logs/calibrate/cmodel.ckpt \
  experiment=area_curv_stab_coco-person_yuhi \
  data.eval_source=trainval \
  data.eval_subset=cal \
  max_images=null \
  alpha_steps=9000 \
  w=9 epsilon=0.00009 rho=0.05 \
  tau_stab_sweep.enabled=false \
  +progress_log_interval=10

```

7. Create Graph
```
python src/utils/plot_summary_scatter.py \
  --summary_csv logs/area_curve/runs/<timestamp>/area_curves/summary.csv

```

python src/utils/plot_summary_scatter.py \
  --summary_csv logs/area_curv_stab/runs/area_curv_stab_coco-person_yuhi/2026-04-08_16-49-22/area_curves/summary.csv \
  --mode iou_vs_tau_sweep


python src/utils/plot_summary_scatter.py \
  --summary_csv logs/area_curv_stab/runs/area_curv_stab_coco-person_yuhi_steps200/2026-04-06_16-35-07/area_curv_stab/summary.csv \
  --mode iou_vs_tau_sweep
  
 python src/utils/plot_area_curv_stab.py \
  --summary_csv logs/area_curv_stab/runs/area_curv_stab_coco-person_yuhi/2026-04-12_10-42-55/area_curv_stab/summary.csv \
  --mode all_tau \
  --include_base

8. Visualization
```
python src/visualize/visualize.py \
  ckpt_path=/home/chiba/research/kandinsky/kandinsky-calibration/logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel/step20k/cmodel.ckpt \
  image_indices=[0,1,2]

```