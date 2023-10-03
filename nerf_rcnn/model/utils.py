
#
# Some of the code in this file is adapted from torchvision
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/_utils.py
# https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
# 

import math
from typing import Dict, List, Optional, Tuple
# from matplotlib.pyplot import axis
from .rotated_iou.oriented_iou_loss import cal_iou_3d

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union

import roi_align
import numpy as np


def pt(obj):
    if type(obj) is list or type(obj) is tuple:
        shape_list = []
        for i in obj:
            shape_list.append(pt(i))
        return shape_list
    else:
        return obj.shape
        

def print_shape(obj):
    shape = pt(obj)
    print(shape)


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # print('highest_quality_foreach_gt', highest_quality_foreach_gt)

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


# Reference: https://blog.csdn.net/hxxjxw/article/details/122629725
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        # Pick the largest box and remove any overlap
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        # Compute IoU of the picked box with the rest
        iou = box_iou_3d(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze()
        # Remove boxes with an IoU above the threshold
        idxs = idxs[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 6] or Tensor[N, 7]): 
            - If Tensor[N, 6]: AABB boxes in ``(x1, y1, z1, x2, y2, z2)`` format
                with ``0 <= x1 < x2``, ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
            - If Tensor[N, 7]: OBB boxes in ``(x, y, z, w, h, d, theta)`` format
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.

    Args:
        boxes (Tensor[N, 6] or Tensor[N, 7]): 
            - If Tensor[N, 6]: AABB boxes in ``(x1, y1, z1, x2, y2, z2)`` format
                with ``0 <= x1 < x2``, ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
            - If Tensor[N, 7]: OBB boxes in ``(x, y, z, w, h, d, theta)`` format
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have all sides
        larger than min_size
    """
    if boxes.size(1) == 6:
        ws, hs, ds = boxes[:, 3] - boxes[:, 0], boxes[:, 4] - boxes[:, 1], boxes[:, 5] - boxes[:, 2]
    elif boxes.size(1) == 7:
        ws, hs, ds = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    keep = (ws >= min_size) & (hs >= min_size) & (ds >= min_size)
    keep = torch.where(keep)[0]
    return keep


def cuboid_tranlate(boxes):
    """
    Translate the 7D representation of oriented bounding box into 8-point version 

    Args:
      boxes (Tensor[N, 7]): (x, y, z, w, h, d, theta)
    Returns:
        _type_: _description_
    """
    all_bounding_box= []
    N = boxes.size(0)
    base_bounding = np.array([[.5, .5, .5], 
                            [.5, -.5, .5],
                            [-.5, -.5, .5],
                            [-.5, .5, .5],
                            [.5, .5, -.5], 
                            [.5, -.5, -.5],
                            [-.5, -.5, -.5],
                            [-.5, .5, -.5],])
    base_bounding = base_bounding.T
    base_bounding = base_bounding.repeat([N, 1], axis=0)
    # print_shape(base_bounding)
    return

    for i in dict['bounding_boxes']:
        extents = np.array(i['extents'])
        orientation = np.array(i['orientation'])
        position = np.array(i['position'])
        bounding = extents[:, None] * base_bounding
        bounding = orientation @ bounding
        bounding = bounding + position[:, None]
        bounding = bounding.T
        # bounding = bounding[[0,4]]
        all_bounding_box.append(bounding)
    return 


def clip_boxes_to_mesh(boxes: Tensor, size: Tuple[int, int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside a mesh of size `size`.

    Args:
        boxes (Tensor[N, _, 6] or Tensor[N, _, 7]): 
            - If Tensor[N, _, 6]: AABB boxes in ``(x1, y1, z1, x2, y2, z2)`` format
                with ``0 <= x1 < x2``, ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
            - If Tensor[N, _, 7]: OBB boxes in ``(x, y, z, w, h, d, theta)`` format
        size (Tuple[width, height, depth]): size of the mesh

    Returns:
        Tensor[N, _, 6] or [N, _, ]: clipped boxes
    """

    width, height, depth = size
    
    if boxes.size(-1) == 6:
        dim = boxes.dim()
        boxes_x = boxes[..., 0::3]
        boxes_y = boxes[..., 1::3]
        boxes_z = boxes[..., 2::3]
        

        boxes_x = boxes_x.clamp(min=0, max=size[0])
        boxes_y = boxes_y.clamp(min=0, max=size[1])
        boxes_z = boxes_z.clamp(min=0, max=size[2])

        clipped_boxes = torch.stack((boxes_x, boxes_y, boxes_z), dim=dim)
        
        return torch.cat([clipped_boxes.reshape(boxes.shape)], dim = -1)
    
    elif boxes.size(-1) == 7:
        # discard obb with center outside the mesh grid
        # [x, y, z, w, h, d, theta]
        x_valid = (boxes[..., 0] >= 0) & (boxes[..., 0] <= size[0])
        y_valid = (boxes[..., 1] >= 0) & (boxes[..., 1] <= size[1])
        z_valid = (boxes[..., 2] >= 0) & (boxes[..., 2] <= size[2])
        valid = x_valid & y_valid & z_valid

        return boxes[valid]


@torch.no_grad()
def batched_box_iou(boxes1: Tensor, boxes2: Tensor, batch_size=16) -> Tensor:
    '''
    Batchify the box_iou_3d function to avoid OOM issues.
    '''
    ious = []
    num_batches = (boxes1.shape[0] + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        if end_idx > boxes1.shape[0]:
            end_idx = boxes1.shape[0]
        ious.append(box_iou_3d(boxes1[start_idx:end_idx], boxes2))

    return torch.cat(ious, dim=0)


def box_iou_3d(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Distinguish AABB and OBB by the length of second dimension of boxes1 and boxes2:
        - If the second dimension is 6, the boxes are AABBs. 
            Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
            ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 < z2``.
        - If the second dimension is 7, the boxes are OBBs.
            Both sets of boxes are expected to be in ``(x, y, z, w, l, h, theta)`` format. 

    Args:
        boxes1 (Tensor[N, 6]) or (Tensor[N, 7]): first set of boxes
        boxes2 (Tensor[M, 6]) or (Tensor[M, 7]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    if boxes1.size(1) == boxes2.size(1) == 6:
        inter, union = _aabb_inter_union_3d(boxes1, boxes2)
        iou = inter / union
        return iou
    elif boxes1.size(1) == boxes2.size(1) == 7:
        boxes1_rep = boxes1.unsqueeze(1).repeat(1, boxes2.size(0), 1)
        boxes2_rep = boxes2.unsqueeze(0).repeat(boxes1.size(0), 1, 1)
        iou = cal_iou_3d(boxes1_rep.cuda(), boxes2_rep.cuda()).cpu().type(torch.float32)
        return iou
    else:
        raise ValueError("The second dimension of boxes1 and boxes2 should be the same, both 6 or 7. But get {} and {}.".format(boxes1.size(1), boxes2.size(1)))


def aabb_volume(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, z1, x2, y2, z2) coordinates.

    Args:
        boxes (Tensor[N, 6]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, z1, x2, y2, z2) format with
            ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 < z2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _aabb_inter_union_3d(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    volume1 = aabb_volume(boxes1)
    volume2 = aabb_volume(boxes2)


    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whd = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N,M]

    union = volume1[:, None] + volume2 - inter

    return inter, union


def vis_iou_of_anchor_and_gt(anchors, targets, pos_thres=0.7, neg_thres=0.3, abort=False):
    """ Compute the IOU between anchors and ground truth boxes of each mesh.
        This function can only be called in training mode, when `targets` is not None.
        Args:
            anchors (List[Tensor]): anchors for each mesh.
            targets (List[Tensor]): ground truth boxes for each mesh.
            pos_thres (float): threshold for positive anchors.
            neg_thres (float): threshold for negative anchors.
            abort (bool): if True, abort after the function is called. This is used for debugging."""
    
    if targets is None:
        raise ValueError("This function can only be called in training mode, when `targets` is not None.")

    title = "#"*40 + "IOU of anchors and GT" + "#"*40
    print('\n'+title)
    for mesh_idx, (anchor, target) in enumerate(zip(anchors, targets)):
        print("Checking IOU of mesh {}...".format(mesh_idx))
        ious = batched_box_iou(anchor, target, batch_size=512)
        max_iou_per_gt, max_match_idxs = torch.max(ious, axis=0)
        max_iou_per_gt = max_iou_per_gt.detach().cpu().numpy()
        max_match_idxs = max_match_idxs.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        for bbox_idx, (max_match_idx, iou) in enumerate(zip(max_match_idxs, max_iou_per_gt)):
            print('\ttarget={}, iou={:.3f}, best_match={}'.format(target[bbox_idx], iou, anchor[max_match_idx].cpu().detach().numpy()))
        print(f'\tmean_iou={np.mean(max_iou_per_gt)}')

        iou_per_anchor, _ = torch.max(ious, axis=1)
        pos_map = (iou_per_anchor>pos_thres).cpu().detach().numpy() # threshold is hardcoded
        neg_map = (iou_per_anchor<neg_thres).cpu().detach().numpy()
        print(f"\tnum_pos_anchors={np.sum(pos_map)}\n\tnum_neg_anchors={np.sum(neg_map)}")

    print("#"*len(title)+"\n")

    if abort:
        exit()
        
        
def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_x = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
        ex_y = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
        ex_z = ex_rois[:, 5] - ex_rois[:, 2] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_x
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_y
        ex_ctr_z = ex_rois[:, 2] + 0.5 * ex_z

        gt_x = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
        gt_y = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
        gt_z = gt_rois[:, :, 5] - gt_rois[:, :, 2] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_x
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_y
        gt_ctr_z = gt_rois[:, :, 2] + 0.5 * gt_z

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_x
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_y
        targets_dz = (gt_ctr_z - ex_ctr_z.view(1,-1).expand_as(gt_ctr_z)) / ex_z
        targets_dw = torch.log(gt_ctr_x / ex_x.view(1,-1).expand_as(gt_ctr_x))
        targets_dh = torch.log(gt_ctr_y / ex_y.view(1,-1).expand_as(gt_ctr_y))
        targets_dl = torch.log(gt_ctr_z / ex_z.view(1,-1).expand_as(gt_ctr_z))

    elif ex_rois.dim() == 3:
        ex_x = ex_rois[:, :, 3] - ex_rois[:, 0] + 1.0
        ex_y = ex_rois[:, :, 4] - ex_rois[:, 1] + 1.0
        ex_z = ex_rois[:, :, 5] - ex_rois[:, 2] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_x
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_y
        ex_ctr_z = ex_rois[:, :, 2] + 0.5 * ex_z

        gt_x = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
        gt_y = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
        gt_z = gt_rois[:, :, 5] - gt_rois[:, :, 2] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_x
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_y
        gt_ctr_z = gt_rois[:, :, 2] + 0.5 * gt_z

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_x
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_y
        targets_dz = (gt_ctr_z - ex_ctr_z) / ex_z
        targets_dw = torch.log(gt_ctr_x / ex_x)
        targets_dh = torch.log(gt_ctr_y / ex_y)
        targets_dl = torch.log(gt_ctr_z / ex_z)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dl),2)

    return targets




# Modified from torchvision.ops.roi_align
def roi_align_3d(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: Union[Tuple[int, int, int], int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
) -> Tensor:
    """
    A crappier 3D version of the torchvision.ops.roi_align function.
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, W, L, H]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 7] or List[Tensor[L, 6]]): the box coordinates in (x1, y1, z1, x2, y2, z2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 <= z2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (width, length, height).
        spatial_scale (float): a scaling factor that maps the box coordinates to
            the input coordinates. For example, if your boxes are defined on the scale
            of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            the original image), you'll want to set this to 0.5. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1

    Returns:
        Tensor[K, C, output_size[0], output_size[1], output_size[2]]: The pooled RoIs.
    """

    # TODO: make sampling_ratio work
    check_3d_roi_boxes_shape(boxes)
    rois = boxes
    output_size = (output_size, output_size, output_size) if isinstance(output_size, int) else output_size
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)

    roi_inds = rois[:, 0]
    rois = rois[:, 1:]
    roi_inds = roi_inds.contiguous().to(torch.int)

    return roi_align.roi_align.roi_align_3d(input, rois, roi_inds, output_size[0], output_size[1], output_size[2],
                                  spatial_scale)

# From torchvision.ops._utils
def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois

# From torchvision.ops._utils
def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

# Modified from torchvision.ops._utils
def check_3d_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(
                _tensor.size(1) == 6, "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 6]]"
            )
    elif isinstance(boxes, torch.Tensor):
        torch._assert(boxes.size(1) == 7, "The boxes tensor shape is not correct as Tensor[K, 7]")
    else:
        torch._assert(False, "boxes is expected to be a Tensor[L, 7] or a List[Tensor[K, 6]]")
    return

# Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/mask_ops.py
def _do_paste_mask(masks, boxes, img_w: int, img_l: int, img_h: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, W, L, H
        boxes: N, 6, in (x1, y1, z1, x2, y2, z2) format
        img_w, img_l, img_h (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
    Returns:
        if skip_empty == False, a mask of shape (N, img_w, img_l, img_h)
        if skip_empty == True, a mask of shape (N, w', l', h'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int, z0_int = torch.clamp(boxes.min(dim=0).values.floor()[:3] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 4].max().ceil() + 1, max=img_l).to(dtype=torch.int32)
        z1_int = torch.clamp(boxes[:, 5].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int, z0_int = 0, 0, 0
        x1_int, y1_int, z1_int = img_w, img_l, img_h

    x0, y0, z0, x1, y1, z1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_z = torch.arange(z0_int, z1_int, device=device, dtype=torch.float32)
    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32)
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32)
    img_z = (img_z - z0) / (z1 - z0) * 2 - 1
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y, img_z have shapes (N, w), (N, l), (N, h)

    gx = img_x[:, :, None, None].expand(N, img_x.size(1), img_y.size(1), img_z.size(1))
    gy = img_y[:, None, :, None].expand(N, img_x.size(1), img_y.size(1), img_z.size(1))
    gz = img_z[:, None, None, :].expand(N, img_x.size(1), img_y.size(1), img_z.size(1))
    grid = torch.stack([gz, gy, gx], dim=-1)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=True)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(x0_int, x1_int), slice(y0_int, y1_int), slice(z0_int, z1_int))
    else:
        return img_masks[:, 0], ()


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/mask_ops.py
# Annotate boxes as Tensor (but not Boxes) in order to use scripting
@torch.jit.script_if_tracing
def paste_masks_in_image(
    masks: torch.Tensor, boxes: torch.Tensor, image_shape: Tuple[int, int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28 x 28) into an image.
    The location, height, length, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.
    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    Args:
        masks (tensor): Tensor of shape (Bimg, Wmask, Lmask, Hmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = Lmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 6).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): width, length, height
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Wimage, Limage, Himage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    BYTES_PER_FLOAT = 4
    GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

    assert masks.shape[-1] == masks.shape[-2] == masks.shape[-3], "Only cube mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_w, img_l, img_h = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * int(img_l) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in model.utils.paste_masks_in_image is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_w, img_l, img_h, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :, :], boxes[inds], img_w, img_l, img_h, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk

    return img_masks



def mask_iou_3d(masks1: Tensor, masks2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of masks.

    Args:
        masks1 (Tensor[N, W, L, H]): first set of masks
        masks2 (Tensor[M, W, L, H]): second set of masks

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in masks1 and masks2
    """
    masks1 = masks1[:, None, ...].repeat(1, masks2.size(0), 1, 1, 1)
    masks2 = masks2[None, :, ...].repeat(masks1.size(0), 1, 1, 1, 1)
    inter = (masks1 * masks2).sum(dim=(2, 3, 4))
    union = masks1.sum(dim=(2, 3, 4)) + masks2.sum(dim=(2, 3, 4)) - inter
    iou = inter / union
    return iou


if __name__ == '__main__':
    bbox = torch.tensor([1., 1., 1., 1., 1., 1., 0.], dtype = torch.float32)
    cuboid_tranlate(bbox)
    