import json
import numpy as np
import os
from tqdm import tqdm


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


def panoptic_to_semantic(panoptic_seg, seg_info):
    '''
    In the output semantic map, 0 is background, 39 is others, 40 is unlabeled/unknown/void
    '''
    semantic_map = np.zeros_like(panoptic_seg)
    for info in seg_info:
        coco_id = info['category_id']
        if info['isthing']:
            coco_name = coco_things_id_to_name[coco_id]
            if coco_name in coco_things_to_nyu40:
                nyu40_id = coco_things_to_nyu40[coco_name]
            else:
                nyu40_id = 39
        else:
            coco_name = coco_stuff_id_to_name[coco_id]
            if coco_name in coco_stuff_to_nyu40:
                nyu40_id = coco_stuff_to_nyu40[coco_name]
            else:
                nyu40_id = 39

        semantic_id = nyu40_id
        if semantic_id == 40:
            semantic_id = 0 # remap 40 to 0

        semantic_map[panoptic_seg == info['id']] = semantic_id

    # remap 0 to 40
    semantic_map[panoptic_seg == 0] = 40

    return semantic_map


def panoptic_to_instance(panoptic_seg, seg_info):
    instance_map = np.zeros_like(panoptic_seg)
    instance_info = []
    for info in seg_info:
        coco_id = info['category_id']
        if info['isthing']:
            coco_name = coco_things_id_to_name[coco_id]
            if coco_name in coco_things_to_nyu40:
                nyu40_id = coco_things_to_nyu40[coco_name]
            else:
                nyu40_id = 39
        else:
            coco_name = coco_stuff_id_to_name[coco_id]
            if coco_name in coco_stuff_to_nyu40:
                nyu40_id = coco_stuff_to_nyu40[coco_name]
            else:
                nyu40_id = 39

        if nyu40_id == 40:
            continue    # background

        instance = {
            'id': len(instance_info) + 1,
            'nyu40_id': nyu40_id,
        }
        instance_info.append(instance)
        instance_map[panoptic_seg == info['id']] = instance['id']

    return instance_map, instance_info


if __name__ == '__main__':
    panoptic_dir = ''
    semantic_dir = ''
    instance_dir = ''

    scenes = os.listdir(panoptic_dir)
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)

    for scene in tqdm(scenes):
        scene_dir = os.path.join(panoptic_dir, scene)
        images = os.listdir(scene_dir)
        images = [x for x in images if x.endswith('.jpg')]

        out_dir = os.path.join(semantic_dir, scene)
        # out_dir = os.path.join(instance_dir, scene)
        os.makedirs(out_dir, exist_ok=True)

        for image in tqdm(images):
            with open(os.path.join(scene_dir, image.replace('.jpg', '.json')), 'r') as f:
                seg_info = json.load(f)

            panoptic_seg = np.load(os.path.join(scene_dir, image.replace('.jpg', '.npy')))
            semantic_map = panoptic_to_semantic(panoptic_seg, seg_info)
            np.save(os.path.join(out_dir, image.replace('.jpg', '.npy')), semantic_map)

            # instance_map, instance_info = panoptic_to_instance(panoptic_seg, seg_info)
            # np.save(os.path.join(out_dir, image.replace('.jpg', '.npy')), instance_map)
            # with open(os.path.join(out_dir, image.replace('.jpg', '.json')), 'w') as f:
            #     json.dump(instance_info, f, indent=2)
