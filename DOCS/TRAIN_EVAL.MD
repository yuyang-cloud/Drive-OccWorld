# Train and Evaluate

**1. Generate the Inflated Occupancy Annotations as described [here](https://github.com/yuyang-cloud/Drive-OccWorld/blob/c508199c69d6d14e4618f22ad16e2ddb91606b8a/DOCS/DATASET.MD?plain=1#L49):**

1) Set ```only_generate_dataset=True``` in both [projects/configs/inflated/action_condition_GMO.py](../projects/configs/inflated/action_condition_GMO.py) and [projects/configs/inflated/action_condition_MMO.py](../projects/configs/inflated/action_condition_MMO.py)
2) Run the training and evaluation scripts using these configurations. Since the model's forward process is not executed, this will not consume GPU memory:
```bash
CONFIG=projects/configs/inflated/action_condition_GMO.py
GPU_NUM=8   # 8 GPUs(Processes) consume ~20 hours

# Generate training data
./tools/dist_train.sh ${CONFIG} ${GPU_NUM}

# Generate validation data
CKPT=work_dirs/action_condition_GMO/epoch_1.pth
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```


**2. Train**

```bash
CONFIG=path/to/config.py
GPU_NUM=8

./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

Configuration Files:
1) [inflated/action_condition_GMO.py](../projects/configs/inflated/action_condition_GMO.py): Predicts General Moveable Objects (GMO) using the inflated occupancy annotations as defined in Cam4DOcc.
2) [inflated/action_condition_MMO.py](../projects/configs/inflated/action_condition_MMO.py): Predicts Multi-class Moveable Objects (MMO) using the inflated occupancy annotations defined in Cam4DOcc.
3) [fine_grained/action_condition_GMO.py](../projects/configs/fine_grained/action_condition_GMO.py): Predicts General Moveable Objects (GMO) using the fine-grained occupancy annotations defined in nuScenes-Occupancy.
4) [fine_grained/action_condition_GMO_GSO.py](../projects/configs/fine_grained/action_condition_GMO_GSO.py): Predicts General Moveable Objects (GMO) and General Static Objects (GSO) using the fine-grained occupancy annotations defined in nuScenes-Occupancy.
5) [fine_grained/action_condition_MMO_MSO.py](../projects/configs/fine_grained/action_condition_MMO_MSO.py): Predicts Multi-class Moveable Objects (MMO) and Multi-class Static Objects (MSO) using the fine-grained occupancy annotations defined in nuScenes-Occupancy.


**3. Evaluate**

Download the provided pretrained [weights](https://huggingface.co/Jianbiao/IR-WM) or use the weights you have trained yourself.

```bash
CONFIG=path/to/config.py
CKPT=work_dirs/config_file_name/epoch_24.pth
GPU_NUM=8

./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```