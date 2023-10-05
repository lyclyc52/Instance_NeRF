import numpy as np
import cv2
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import json
import argparse

with open('coco_id_to_name.json', 'r') as f:
    coco_class_names = json.load(f)

coco_things_id_to_name = {i: v for i, v in enumerate(coco_class_names['thing_classes'])}
coco_stuff_id_to_name = {i: v for i, v in enumerate(coco_class_names['stuff_classes'])}

# 40 is background, 0 is unlabeled/unknown/void, 39 is others
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

colors = np.multiply([
    plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
], 255).astype(np.uint8)


def read_projections(projections):
    proj = []
    for p in projections:
        img = cv2.imread(p)
        mask = img[:, :, 0] > 0
        proj.append(mask)

    return proj


def convert_seg(panoptic_seg, segments_info):
    assert panoptic_seg.min() >= 0
    panoptic_seg = panoptic_seg.astype(np.int32)
    result = np.zeros_like(panoptic_seg)
    result[panoptic_seg == 0] = -1    # unlabeled

    for seg in segments_info:
        assert seg['id'] > 0
        if seg['isthing']:
            name = coco_things_id_to_name[seg['category_id']]
            if name in coco_things_to_nyu40:
                nyu40_id = coco_things_to_nyu40[name]
            else:
                nyu40_id = 39
        else:
            name = coco_stuff_id_to_name[seg['category_id']]
            if name in coco_stuff_to_nyu40:
                nyu40_id = coco_stuff_to_nyu40[name]
            else:
                nyu40_id = 39

        if nyu40_id == 40:      # background
            result[panoptic_seg == seg['id']] = 0
        else:
            result[panoptic_seg == seg['id']] = seg['id']

    return result


def match_seg(proj_dir, seg_dir, out_dir, iou_thresh=0.05):
    seg_maps = os.listdir(seg_dir)
    seg_maps = [x for x in seg_maps if x.endswith('.npy')]
    seg_maps.sort()

    proj_files = os.listdir(proj_dir)
    proj_files = [x for x in proj_files if x.endswith('.png') and '_' in x]
    proj_files = [x for x in proj_files if x.split('_')[1] != '0.png']
    proj_files.sort()

    os.makedirs(out_dir, exist_ok=True)
    for seg in tqdm(seg_maps):
        seg_map = np.load(os.path.join(seg_dir, seg))
        seg_map = seg_map.astype(np.int32)

        with open(os.path.join(seg_dir, seg.replace('.npy', '.json')), 'r') as f:
            seg_info = json.load(f)
        seg_map = convert_seg(seg_map, seg_info)
        output = np.copy(seg_map)

        img_idx = seg.split('.')[0]
        projs = [x for x in proj_files if x.startswith(img_idx)]
        instance_ids = [int(x.split('_')[1].split('.')[0]) for x in projs]
        projs = [os.path.join(proj_dir, x) for x in projs]

        proj_masks = read_projections(projs)

        if len(proj_masks) > 0:
            ids = np.unique(seg_map)

            for i, id in enumerate(ids):
                if id <= 0:
                    continue
                iou = np.zeros((len(proj_masks), ))
                for j, mask in enumerate(proj_masks):
                    iou[j] = np.sum((seg_map == id) & mask) / np.sum((seg_map == id) | mask)

                max_iou = np.max(iou)
                max_idx = np.argmax(iou)
                if max_iou > iou_thresh:
                    output[seg_map == id] = instance_ids[max_idx]
                else:
                    output[seg_map == id] = -1
        else:
            output[seg_map > 0] = -1

        np.save(os.path.join(out_dir, seg), output)

        with h5py.File(os.path.join(out_dir, seg.replace('.npy', '.hdf5')), 'w') as file:
            file.create_dataset('cp_instance_id_segmaps', data=output)

        image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for id in np.unique(output):
            color = colors[id % 37] if id >= 0 else [0, 0, 0]
            image[output == id] = color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, seg.replace('.npy', '.png')), image)


def get_parser():
    parser = argparse.ArgumentParser(description="match mask2former masks with projected masks")

    parser.add_argument('--proj_dir', type=str, help='directory of projected masks')
    parser.add_argument('--seg_dir', type=str, help='directory of mask2former masks')
    parser.add_argument('--out_dir', type=str, help='output directory')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    scenes = os.listdir(args.seg_dir)
    scenes.sort()
    os.makedirs(args.out_dir, exist_ok=True)

    for scene in scenes:
        match_seg(os.path.join(args.proj_dir, scene), 
            os.path.join(args.seg_dir, scene), os.path.join(args.out_dir, scene), 
            iou_thresh=0.05)
