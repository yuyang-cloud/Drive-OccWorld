# Installation

The installation step is similar to [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md).
For convenience, we list the steps below:

**1. Create a conda virtual environment and activate it.**
```shell
conda create -n driveoccworld python=3.8 -y
conda activate driveoccworld
```

**2. Install PyTorch and torchvision (torch<1.11.0 which is required by mmdet3d).**
```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

**3. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # Make sure the GCC version compatible with CUDA
```

**4. Install some other required packges and Detectron2.**
```bash
pip install setuptools einops fvcore seaborn ninja iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 casadi==3.5.5 pytorch-lightning==1.2.5 lyft_dataset_sdk nuscenes-devkit plyfile networkx==2.2 trimesh==2.35.39 yapf==0.40.1

# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# or install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

**5. Install mm-series packages.**
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# Install mmdetection3d from source codes.
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install 
```

**6. Clone repository and setup.**
```bash
git clone https://github.com/yuyang-cloud/Drive-OccWorld.git
cd Drive-OccWorld
git checkout ir-wm

mkdir pretrained
cd pretrained & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth

# Install DCNv3
cd ../projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
python setup.py install
```