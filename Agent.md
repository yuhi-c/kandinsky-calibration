# For AI Agent

## GPU environment
There are two GPU in my environment.
[DETAIL]
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   39C    P0    28W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:5E:00.0 Off |                    0 |
| N/A   33C    P0    27W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

If you wanna run specific GPU, use CUDA_VISIBLE_DEVICES=1 command. By using this command, you can use 2GPU at a same time, so can work more efficiently.

## How to be in excution environment
run
```
conda activate kandinsky-train
```
And you will be in the excution venv

## Running command
Before running the connand, you better set 
```
export DATA_TRAINVAL_ROOT=/home/chiba/research/kandinsky/kandinsky-data/coco/trainval
export DATA_TEST_ROOT=/home/chiba/research/kandinsky/kandinsky-data/coco/test
```
For any command if you get error

- training
```
python src/train.py experiment=train_coco-person_yuhi
```

- calibration
```
python src/calibrate.py experiment=cal_coco-person_t1000_c2000_pixel_yuhi ckpt_path=logs/train/runs/train_coco-person_yuhi/<latest time stamp>/checkpoints/last.ckpt
```

- curve creating
```
python src/area_curve.py ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel_yuhi/2026-04-05_15-55-10/cmodel.ckpt
```
To get only stab, use area_curv_stab.py, to get stab and its pearson, use area_curv_stab_PearsonOnly.py
To set high param, use something like this
```
python src/area_curve.py \
  ckpt_path=logs/calibrate/runs/cal_coco-person_t1000_c2000_pixel_yuhi/<latest time stamp>/cmodel.ckpt \
  experiment=area_curve_coco-person_yuhi \
  'tau_stab_sweep.enabled=true' \
  'tau_stab_sweep.w_values=[1,2,3,4,5]' \
  'tau_stab_sweep.epsilon_values=[0.001,0.005,0.01]' \
  'tau_stab_sweep.rho_values=[0.05,0.1,0.2]'
```