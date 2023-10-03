# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
# Further modified for Instance NeRF mask acquisition
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

# from detectron2.data import MetadataCatalog
# thing_classes = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg").thing_classes
# stuff_classes = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg").stuff_classes

# with open('coco_id_to_name.json', 'w') as f:
#     json.dump({
#         'thing_classes': thing_classes,
#         'stuff_classes': stuff_classes
#     }, f, indent=2)

CLASS_IGNORE = ['other', 'wall', 'floor', 'building', 'pavement', 'mountatin', 'grass', 'dirt',
    'window', 'ceiling', 'tree', 'door', 'roof', 'sea', 'sand', 'snow', 'road', 'river']

NYU40_CLASS_NAMES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
    'pillow', 'mirror', 'floormat', 'clothes', 'ceiling', 'books',
    'refrigerator', 'television', 'paper', 'towel', 'showercurtrain', 'box',
    'whiteboard', 'person', 'nightstand', 'toilet', 
    'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]

coco_things_to_nyu40 = {
    'chair': 5, 
    'couch': 6,
    'bed': 4,
    'dining table': 7,
}

coco_stuff_to_nyu40 = {
    'chair': 5,
    'couch': 6,
    'bed': 4,
    'dining table': 7,
    'curtain': 40,
    'door-stuff': 40,
    'floor-wood': 40,
    'light': 35,
    'shelf': 10,
    'stairs': 40,
    'wall-brick': 40,
    'wall-stone': 40,
    'wall-tile': 40,
    'wall-wood': 40,
    'window-blind': 40,
    'window-other': 40,
    'ceiling-merged': 40,
    'cabinet-merged': 3,
    'table-merged': 7,
    'floor-other-merged': 40,
    'building-other-merged': 40,
    'wall-other-merged': 40,
}


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def save_seg_results(panoptic_seg, segments_info, out_path):
    panoptic_seg = panoptic_seg.cpu().numpy()
    np.save(out_path, panoptic_seg)
    with open(out_path.replace('.npy', '.json'), 'w') as f:
        json.dump(segments_info, f, indent=2)

    return

    for seg in segments_info:
        assert seg['category_id'] > 0
        if seg['isthing']:
            name = thing_classes[seg['category_id']]
            if name in coco_things_to_nyu40:
                nyu40_id = coco_things_to_nyu40[name]
            else:
                nyu40_id = 39
        else:
            name = stuff_classes[seg['category_id']]
            if name in coco_stuff_to_nyu40:
                nyu40_id = coco_stuff_to_nyu40[name]
            else:
                nyu40_id = 39

        nyu40_name = NYU40_CLASS_NAMES[nyu40_id - 1]
        print(f'{name} -> {nyu40_name} ({nyu40_id})')

        # for i in CLASS_IGNORE:
        #     if i in name:
        #         seg['category_id'] = 0
        #         break

    assert panoptic_seg.min() >= 0
    panoptic_seg = panoptic_seg.astype(np.int32)
    # panoptic_seg[panoptic_seg == 0] = -1

    for seg in segments_info:
        if seg['category_id'] == 0:
            panoptic_seg[panoptic_seg == seg['id']] = 0
        else:
            panoptic_seg[panoptic_seg == seg['id']] = seg['id']

    np.save(out_path, panoptic_seg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

                if "panoptic_seg" in predictions:
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    save_seg_results(panoptic_seg, segments_info, out_filename.replace('.jpg', '.npy'))

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
