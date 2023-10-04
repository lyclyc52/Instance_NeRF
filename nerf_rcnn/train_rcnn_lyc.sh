#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/disk1/yliugu/3dfront_mask_data_release

python3 -u run_fcos.py \
--mode train \
--dataset front3d \
--resolution 160 \
--backbone_type vgg_EF \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/metadata \
--dataset_split ${DATA_ROOT}/dataset_split.json \
--save_path /disk1/yliugu/instance_nerf_release/nerf_rpn \
--num_epochs 160 \
--lr 3e-4 \
--weight_decay 1e-3 \
--clip_grad_norm 0.1 \
--log_interval 30 \
--eval_interval 4 \
--keep_checkpoints 2 \
--norm_reg_targets \
--centerness_on_reg \
--center_sampling_radius 1.5 \
--iou_loss_type iou \
--rot_scale_prob 0.0 \
--log_to_file \
--nms_thresh 0.3 \
--batch_size 4 \
--gpus 0-3 \
--wandb
