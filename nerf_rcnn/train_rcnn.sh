#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/path/to/dataset/root

python3 -u run_rcnn.py \
--mode train \
--rpn_type fcos \
--backbone_type vgg_EF \
--dataset_root ${DATA_ROOT} \
--dataset_split ${DATA_ROOT}/dataset_split.json \
--rpn_ckpt /path/to/nerf_rpn/model.pt \
--save_path /path/to/output/folder \
--num_epochs 200 \
--lr 1e-3 \
--weight_decay 1e-2 \
--clip_grad_norm 0.1 \
--log_interval 20 \
--eval_interval 5 \
--keep_checkpoints 2 \
--rpn_head_conv_depth 4 \
--rpn_batch_size_per_mesh 256 \
--rpn_pre_nms_top_n_test 2500 \
--rpn_post_nms_top_n_test 2500 \
--rpn_nms_thresh 0.3 \
--log_to_file \
--bbox_type aabb \
--use_input_rois \
--batch_size 16 \
--gpus 0-3
