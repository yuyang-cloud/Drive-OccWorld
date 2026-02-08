# Prepare Dataset

**Folder structure**
```
Drive-OccWorld
├── projects/
├── tools/
├── pretrained/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train_new.pkl
|   |   ├── nuscenes_infos_temporal_val_new.pkl
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   ├── nuScenes-Occupancy/
```

**1. Download nuScenes V1.0 full dataset and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running:**
```bash
cd Drive-OccWorld
mkdir data
ln -s /path/to/nuscenes data/nuscenes
ln -s /path/to/nuscnens/can_bus data/can_bus
```

**2. Prepare nuscene.pkl file**
1) Download the pre-processed [nuscenes_infos_temporal_train_new.pkl](https://drive.google.com/file/d/1rjyqgJAnjg4Iq1MSussm5yhKDPVIlIxe/view?usp=sharing) and [nuscenes_infos_temporal_val_new.pkl](https://drive.google.com/file/d/1UsIaoX6uw2am9FWJBRA5cR40hl8AcL7p/view?usp=sharing)
2) Move the ```.pkl``` files into the nuscenes folder:
```bash
mv nuscenes_infos_temporal_train_new.pkl /path/to/nuscenes
mv nuscenes_infos_temporal_val_new.pkl /path/to/nuscenes
```

**3. Generate the inflated-occupancy annotations as defined in [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc):**

Create the dataset folders, which will automatically generate data during the first epoch of training and testing:
```bash
mkdir /path/to/save_data/cam4docc   # ~650G
cd /path/to/save_data/cam4docc
mkdir GMO GMO_lyft MMO MMO_lyft
ln -s /path/to/save_data/cam4docc Drive-OccWorld/data
```

**4. Download the fine-grained occupancy annotations as defined in [nuScenes-Occupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md). Then organize the dataset as follows:**
```bash
mv nuScenes-Occupancy-v0.1.7z /path/to/save_data
cd /path/to/save_data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
ln -s /path/to/save_data/nuScenes-Occupancy Drive-OccWorld/data/
```