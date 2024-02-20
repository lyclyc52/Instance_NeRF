## Acquire Initial Masks for Instance-NeRF using Mask2Former

We provide a modified command line tool from [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/demo) to get the initial masks used to train an instance field.
The example usage is given in `run_mask2former.sh`. Before you use the sample code, you need to first install Mask2Former according to the [instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) of their original repo. You will need to download the pretrained weights of Mask2Former [here](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#panoptic-segmentation). The model we are using is [Mask2Former Swin-L (IN21k)](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl). Then you can copy the relevant files to the `demo` folder and run the sample code.

After you get both the Mask2Former masks and the NeRF-RCNN discrete 3D masks, you can run `match_seg.py` to get the matched multi-view consistent masks ready for instance field training:
```bash
python match_seg.py \
--proj_dir path/to/projected/masks \
--seg_dir path/to/mask2former/masks \
--out_dir path/to/output/matched/masks
```
