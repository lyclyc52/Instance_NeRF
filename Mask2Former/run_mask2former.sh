python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input /path/to/input/*.jpg \
  --output /path/to/output \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS ../swin_l_panoptic_seg.pkl
