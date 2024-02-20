# Instance-NeRF
[![arXiv](https://img.shields.io/badge/arXiv-2304.04395-f9f107.svg)](https://arxiv.org/abs/2304.04395) [![Youtube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=wW9Bme73coI) [<img src="https://img.shields.io/badge/Cite-BibTex-orange">](#citation)

Instance Neural Radiance Field [Instance-NeRF, ICCV 2023].

This is the official PyTorch implementation of [Instance-NeRF](https://arxiv.org/abs/2304.04395).

> [**Instance Neural Radiacne Field**](https://arxiv.org/abs/2304.04395)           
> Yichen Liu*, Benran Hu*, Junkai Huang*, Yu-Wing Tai, Chi-Keung Tang           
> IEEE/CVF International Conference on Computer Vision (ICCV), 2023     
> *\* indicates equal contribution*


## Instance-NeRF Model Architecture
<img src="imgs/main.png" width="830"/>


## Installation

First, clone this repo and the submodules.
```bash
git clone https://github.com/lyclyc52/Instance_NeRF.git --recursive
```

Ther are two submodules used in the repo:
- [RoIAlign.pytorch](https://github.com/zymk9/RoIAlign.pytorch/tree/b0fa4dbe45a21b2573275965bdeee1f0a3a9b326): It will be used in the NeRF-RCNN training. We adapt the 2D RoIAlign to 3D input.
- [torch-ngp](https://github.com/zymk9/torch-ngp/tree/instance_nerf): We modified torch-ngp to add instance field training.


To install Instance-NeRF:

1. Create a conda environment:
```bash
conda env create -f environment.yml
conda activate instance_nerf
```

2. Follow the instructions in [RoIAlign.PyTorch](https://github.com/zymk9/RoIAlign.pytorch/tree/b0fa4dbe45a21b2573275965bdeee1f0a3a9b326) and [torch-ngp](https://github.com/zymk9/torch-ngp/tree/instance_nerf) to compile the extensions and install related packages.


## Train Instance-NeRF
An overview of the entire training process:
1. Train NeRF models of the scenes and extract the RGB and density.
2. Train a NeRF-RCNN model using the extracted features and 3D annotations.
3. Perform inference on unseen NeRF scenes to get discrete 3D masks.
4. Run [Mask2Former](https://github.com/facebookresearch/Mask2Former) to get the initial 2D segmentation masks of the scenes. Use the 3D masks to match 2D masks.
5. Train an instance field with the masks aligned, and optionally refine the NeRF-produced masks with CascadePSP and repeat NeRF training.

For step 1-3, please refer to the documentation in [nerf_rcnn](nerf_rcnn/README.md).

For step 4-5, please check the docs in [instance_nerf](https://github.com/zymk9/torch-ngp/tree/instance_nerf#instance-field-training).


## Inference 
We provide an example to use our code.
1. Create the [envirnoment](##Installation), download the [dataset](##Dataset) and the checkpoint of [NeRF-RCNN](###NeRF-RCNN)
2. Predict the coarse 3D mask using the sample script [here](./nerf_rcnn/inference.sh)
3. Download the [NeRF training data](https://hkustconnect-my.sharepoint.com/personal/yliugu_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fyliugu%5Fconnect%5Fust%5Fhk%2FDocuments%2FInstance%5FNeRF%5FData%2F3DFRONT%5Fdata%2Fnerf%5Fdata). Instance field is a scene-specific model so you only need to download the scenes you want here.
The following insturctions are under [instance_nerf](https://github.com/zymk9/torch-ngp/tree/6be6af198f1092e8d75574727a030ae15e199fe8) repo.
4. Train the NeRF model using the [sample script](./instance_nerf/README.md###nerf-training)
5. Prepare masks following the [Mask Preparation Section](https://github.com/zymk9/torch-ngp/tree/6be6af198f1092e8d75574727a030ae15e199fe8?tab=readme-ov-file#mask-preparation). Basically, it contains three steps:
   - Produce 2D instance segmentation masks by Mask2Former using this [sample code](./Mask2Former_sample/run_mask2former.py). The detailed instructions are in this [README](./Mask2Former_sample/README.md) under this repo.
   - Produce 2D projected segmentation masks. You can find the script under [our torch-ngp repo](https://github.com/zymk9/torch-ngp/blob/6be6af198f1092e8d75574727a030ae15e199fe8/scripts/project_3d_masks.py)
   - Match these 2D masks using this [sample code](./Mask2Former_sample/match_seg.py) under this repo.
6. Train the instance NeRF following the [Instance Field Inference section](https://github.com/zymk9/torch-ngp/tree/6be6af198f1092e8d75574727a030ae15e199fe8?tab=readme-ov-file#instance-field-inference) under our torch-ngp repo



## Pre-trained Weights

### NeRF-RCNN 

You can download our pre-trained NeRF-RPN and NeRF-RCNN models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yliugu_connect_ust_hk/EiAyN_I_coZDh_gUjH8_wDkBjhVWefZ26cP35bovIrxwWA?e=ZWOxyW). 

To train from scratch, first you need to train a NeRF-RPN model. It is based on [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN/tree/main/nerf_rpn) and we disable the `--rotated_box` flag. 
We provide sample training and testing shell scripts called `train/test_rpn/rcnn.sh` for NeRF-RPN and NeRF-RCNN under [nerf_rcnn](./nerf_rcnn) folder, 


## Dataset
We extended the 3D-FRONT NeRF Dataset used in [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN) by increasing the number of scenes from ~250 to ~1k, adding instance labels for each object, as well as including 2D and 3D instance segmentation masks. The entire dataset we used for training is availible [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yliugu_connect_ust_hk/EmoTMwuZXnNCoe7Yi7PRBQQBi86oHNd2CWDMwsy1ZdwsDA?e=siHQNU).

For training Instance-NeRF, as well as NeRF-RCNN on your custom datasets, please refer to both the [NeRF-RPN dataset creation](https://github.com/lyclyc52/NeRF_RPN/blob/main/data/README.md#nerf-rpn-dataset) and the [NeRF-RCNN training guide](https://github.com/hjk0918/NeRF_RCNN/tree/public_version/nerf_rcnn#nerf-rcnn-training).


### NeRF-RCNN Dataset Creation

We build our dataset based on [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).

If you want to know more details, you can refer to this forked [BlenderProc repo](https://github.com/hjk0918/BlenderProc/tree/main/scripts) for how we generate our data. For NeRF training and feature extraction, please check [this repo](https://github.com/zymk9/instant-ngp/tree/master/scripts). To predict RoIs, please check [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN/tree/main).

**Note**: The pre-trained NeRF-RPN model we published here is different from that in NeRF-RPN repo. In this paper, we use AABB bounding boxes and include more data in our training. 


## Citation
If you find Instance-NeRF useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{instancenerf,
    title = {Instance Neural Radiance Field},
    author = {Liu, Yichen and Hu, Benran and Huang, Junkai and Tai, Yu-Wing and Tang, Chi-Keung},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year = {2023}
}
```
