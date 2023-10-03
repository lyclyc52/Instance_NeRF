
#
# The AP and recall metrics are adapted from Detectron:
# https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/voc_eval.py
# https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/json_dataset_evaluator.py
#


import math
import torch
from collections import defaultdict
from model.utils import box_iou_3d, mask_iou_3d, print_shape


def evaluate_box_proposals_recall(proposals_list, proposal_scores_list, gt_boxes_list, thresholds=None, limit=None):
    """
    Evaluate detection proposal recall metrics.
    """

    gt_overlaps = []
    num_pos = 0

    for proposals, scores, gt_boxes in zip(proposals_list, proposal_scores_list, gt_boxes_list):

        ids = torch.argsort(scores, descending=True)
        proposals = proposals[ids]
        scores = scores[ids]

        if proposals.shape[0] == 0 or gt_boxes.shape[0] == 0:
            continue

        num_pos += gt_boxes.shape[0]

        if limit is not None and len(proposals) > limit:
            proposals = proposals[:limit]

        overlaps = box_iou_3d(proposals, gt_boxes)
        _gt_overlaps = torch.zeros(gt_boxes.shape[0])
        
        for j in range(min(proposals.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # print(max_overlaps)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_box_proposals_precision(proposals_list, proposal_scores_list, gt_boxes_list, 
                                     score_thresh=0.0, thresholds=None, limit=None):
    """
    Evaluate detection proposal precision with a given confidence threshold 
    and an IoU threshold
    """

    box_overlaps = []
    num_det = 0

    for proposals, scores, gt_boxes in zip(proposals_list, proposal_scores_list, gt_boxes_list):

        # Filter out low scoring boxes
        ids = scores >= score_thresh
        ids = ids.nonzero(as_tuple=True)
        proposals = proposals[ids]
        scores = scores[ids]

        ids = torch.argsort(scores, descending=True)
        proposals = proposals[ids]
        scores = scores[ids]

        if proposals.shape[0] == 0:
            continue

        # Only consider the first k detections.
        if limit is not None and len(proposals) > limit:
            proposals = proposals[:limit]

        num_det += proposals.shape[0]

        overlaps = box_iou_3d(proposals, gt_boxes)
        _box_overlaps = torch.zeros(proposals.shape[0])
        
        for j in range(min(proposals.shape[0], gt_boxes.shape[0])):
            # find which gt box maximally covers each proposal box
            # and get the iou amount of coverage for each proposal box
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # find which proposal box is 'best' covered (i.e. 'best' = most iou)
            box_ovr, box_ind = max_overlaps.max(dim=0)
            assert box_ovr >= 0
            # find the gt box that covers the best covered proposal box
            gt_ind = argmax_overlaps[box_ind]
            # record the iou coverage of this proposal box
            _box_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _box_overlaps[j] == box_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        box_overlaps.append(_box_overlaps)

    box_overlaps = (
        torch.cat(box_overlaps, dim=0) if len(box_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    box_overlaps, _ = torch.sort(box_overlaps)

    # IoU thresholds
    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)

    precisions = torch.zeros_like(thresholds)
    # compute precision for each iou threshold
    for i, t in enumerate(thresholds):
        precisions[i] = (box_overlaps >= t).float().sum() / float(num_det)
    
    # Average precision vs IoU
    ap = precisions.mean()
    return {
        "ap": ap,
        "precisions": precisions,
        "thresholds": thresholds,
        "score_thresh": score_thresh,
        "box_overlaps": box_overlaps,
        "num_det": num_det,
    }

   
def evaluate_labels(proposals_list, gt_boxes_list, AP_threshold = [0.25, 0.5]):
    
    gt_labels_list = []
    for threshold in AP_threshold:
        cur_gt_labels_list = []
        for cur_proposals, cur_gt_bboxes in zip(proposals_list, gt_boxes_list):
            
            overlaps = box_iou_3d(cur_proposals, cur_gt_bboxes)  
            max_overlaps, _ = overlaps.max(dim=1)
            cur_gt_labels = torch.zeros(max_overlaps.size(0)).type(torch.int32)
            positive_index = torch.nonzero(max_overlaps >= threshold)
            cur_gt_labels[positive_index] = 1
            cur_gt_labels_list.append(cur_gt_labels)
        gt_labels_list.append(cur_gt_labels_list)
    return gt_labels_list


def evaluate_classificaiton_accuracy(scores_list, gt_label_list, threshold):
    """
    Evaluate detection proposal accuracy metrics.
    """
    acc = []
    for i in range(len(scores_list)):
        cur_score = scores_list[i]
        cur_gt_label = gt_label_list[i]
        
        
        objectness_index = (cur_score > threshold).type_as(cur_gt_label)
        
        match = (cur_gt_label == objectness_index).type(torch.float)
        cur_acc = torch.sum(match) / match.size(0)
        acc.append(cur_acc)
    
    return sum(acc) / len(acc)


def evaluate_classificaiton(scores_list, gt_label_list, threshold):
    precisions = []
    accurcy = []
    precision_100 = []
    for cur_score, cur_gt_label in zip(scores_list, gt_label_list):
        if len(cur_score.shape) > 1:
            cur_score = cur_score[..., 1]
        sort_index = cur_score.sort(descending=True)[1]
        sort_index = sort_index[:100]
        cur_precision_100 = cur_gt_label[sort_index].sum() / cur_gt_label[sort_index].size(0)
        precision_100.append(cur_precision_100)

        positive_mask = (cur_score> threshold)
        positive_index = cur_gt_label[torch.nonzero(positive_mask)]
        cur_precision = torch.sum(positive_index) / positive_index.size(0)
        if positive_index.size(0) > 0:
            precisions.append(cur_precision)
        
        match = (cur_gt_label == positive_mask.type_as(cur_gt_label)).type(torch.float)
        cur_accurcy = torch.sum(match) / match.size(0)
        accurcy.append(cur_accurcy)
    
    return {'precision': sum(precisions) / len(precisions) if len(precisions)>0 else 0,
            'accurcy': sum(accurcy) / len(accurcy) if len(accurcy)>0 else 0,
            'precision_100': sum(precision_100) / len(precision_100) if len(precision_100)>0 else 0 }


def evaluate_box_proposals_average_precision(proposals_list, proposal_scores_list, gt_boxes_list, 
                                             iou_thresh=0.25, top_k=None):
    """
    Evaluate detection average precision with a given IoU threshold
    """

    box_overlaps = []
    box_scores = []
    num_gt = 0

    for proposals, scores, gt_boxes in zip(proposals_list, proposal_scores_list, gt_boxes_list):

        if len(scores.shape) > 1:
            scores = scores[..., 1]
        ids = torch.argsort(scores, descending=True)

        proposals = proposals[ids]
        scores = scores[ids]
        num_gt += gt_boxes.shape[0]

        if proposals.shape[0] == 0:
            continue

        # Only consider the first k detections.
        if top_k is not None and len(proposals) > top_k:
            proposals = proposals[:top_k]
            scores = scores[:top_k]

        overlaps = box_iou_3d(proposals, gt_boxes)
        _box_overlaps = torch.zeros(proposals.shape[0])
        _box_scores = torch.zeros(proposals.shape[0])
        
        for j in range(min(proposals.shape[0], gt_boxes.shape[0])):
            # find which gt box maximally covers each proposal box
            # and get the iou amount of coverage for each proposal box
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # find which proposal box is 'best' covered (i.e. 'best' = most iou)
            box_ovr, box_ind = max_overlaps.max(dim=0)
            assert box_ovr >= 0
            # find the gt box that covers the best covered proposal box
            gt_ind = argmax_overlaps[box_ind]
            # record the iou coverage of this proposal box
            _box_overlaps[j] = overlaps[box_ind, gt_ind]
            _box_scores[j] = scores[box_ind]
            assert _box_overlaps[j] == box_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level and scores
        box_overlaps.append(_box_overlaps)
        box_scores.append(_box_scores)

    box_overlaps = (
        torch.cat(box_overlaps, dim=0) if len(box_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    box_scores = (
        torch.cat(box_scores, dim=0) if len(box_scores) else torch.zeros(0, dtype=torch.float32)
    )

    # Confidence thresholds
    step = 0.01
    conf_thresh = torch.arange(0.01, 0.99 + 1e-5, step, dtype=torch.float32)
    precisions = torch.zeros_like(conf_thresh)
    recalls = torch.zeros_like(conf_thresh)
    num_dets = torch.zeros_like(conf_thresh)

    # compute precision and recall for each confidence threshold
    for i, t in enumerate(conf_thresh):
        num_dets[i] = (box_scores >= t).float().sum()
        precisions[i] = (box_overlaps[box_scores >= t] >= iou_thresh).float().sum() / num_dets[i] if num_dets[i]>0 else 0
        recalls[i] = (box_overlaps[box_scores >= t] >= iou_thresh).float().sum() / num_gt if num_gt>0 else 0
    
    # AP from precision-recall curve
    ap = 0

    for i in range(len(precisions) - 1):
        ap += (recalls[i] - recalls[i + 1]) * precisions[i]

    return {
        "ap": ap,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": iou_thresh,
        "score_thresh": conf_thresh,
        "box_overlaps": box_overlaps,
        "num_det": num_dets,
    }


def evaluate_box_proposals_ap(proposals_list, proposal_scores_list, gt_boxes_list, iou_thresh=0.25, top_k=None):
    """
    Evaluate detection average precision with a given IoU threshold,
    using Pascal VOC's AP calculation method.
    """

    num_gt = 0

    scene_ids = []
    all_dets = []
    all_scores = []

    for i, (proposals, scores, gt_boxes) in enumerate(zip(proposals_list, proposal_scores_list, gt_boxes_list)):
        if top_k is not None and len(proposals) > top_k:
            ids = torch.argsort(scores, descending=True)[:top_k]
            proposals = proposals[ids]
            scores = scores[ids]

        scene_ids.extend([i] * len(proposals))
        all_dets.append(proposals)
        all_scores.append(scores)
        num_gt += gt_boxes.shape[0]

    scene_ids = torch.tensor(scene_ids, dtype=torch.int64)
    all_dets = torch.cat(all_dets, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    ids = torch.argsort(all_scores, descending=True)
    all_dets = all_dets[ids]
    all_scores = all_scores[ids]
    scene_ids = scene_ids[ids]
    gt_used = [torch.zeros(len(gt_boxes), dtype=torch.bool) for gt_boxes in gt_boxes_list]

    tp = torch.zeros(len(all_dets), dtype=torch.bool)
    fp = torch.zeros(len(all_dets), dtype=torch.bool)

    for i in range(len(all_dets)):

        overlaps = box_iou_3d(all_dets[i].unsqueeze(0), gt_boxes_list[scene_ids[i]])
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        if max_overlaps > iou_thresh:
            if not gt_used[scene_ids[i]][argmax_overlaps]:
                tp[i] = 1
                gt_used[scene_ids[i]][argmax_overlaps] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)
    recalls = tp / num_gt
    precisions = tp / (tp + fp)

    # AP calculation from PASCAL VOC
    # first append sentinel values at the end
    mrec = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))

    # compute the precision envelope
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return {
        "ap": ap,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": iou_thresh,
        "num_det": tp + fp,
    }


def evaluate_map_recall(pred_boxes_list, pred_scores_list, pred_labels_list,
                        gt_boxes_list, gt_labels_list, iou_thresh=0.25, top_k=None,
                        iou_type='box'):
    """
    Evaluate detection average precision and recall with a given IoU threshold,
    using Pascal VOC's AP calculation method.
    """

    assert iou_type in ['box', 'mask']
    iou_fn = box_iou_3d if iou_type == 'box' else mask_iou_3d
    
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for i, (pred_boxes, scores, pred_labels, gt_boxes, gt_labels) in enumerate(
            zip(pred_boxes_list, pred_scores_list, pred_labels_list, gt_boxes_list, gt_labels_list)):
        
        if top_k is not None and len(pred_boxes) > top_k:
            ids = torch.argsort(scores, descending=True)[:top_k]
            pred_boxes = pred_boxes[ids]
            scores = scores[ids]
            pred_labels = pred_labels[ids]
            
        for l in torch.unique(torch.cat((pred_labels, gt_labels)).to(torch.int64)):
            l = l.item()
            pred_mask_l = pred_labels == l
            pred_bbox_l = pred_boxes[pred_mask_l]
            pred_score_l = scores[pred_mask_l]
            # sort by score
            order = torch.argsort(pred_score_l, descending=True)
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_labels == l
            gt_bbox_l = gt_boxes[gt_mask_l]
            
            n_pos[l] += gt_bbox_l.shape[0]
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            iou = iou_fn(pred_bbox_l, gt_bbox_l)

            gt_overlaps, gt_index = iou.max(dim=1)
            # set -1 if there is no matching ground truth
            gt_index[gt_overlaps < iou_thresh] = -1
            del iou

            selec = torch.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
            
    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = torch.tensor(score[l])
        match_l = torch.tensor(match[l])

        order = torch.argsort(score_l, descending=True)
        match_l = match_l[order]

        tp = torch.cumsum(match_l == 1, dim=0).to(torch.float32)
        fp = torch.cumsum(match_l == 0, dim=0).to(torch.float32)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    ap = torch.zeros(n_fg_class)
    recalls = torch.zeros(n_fg_class)
    for l in range(n_fg_class):
        if rec[l] is not None and rec[l].size(0) != 0:
            recalls[l] = rec[l][-1]
        else:
            recalls[l] = torch.nan

        if prec[l] is None or rec[l] is None:
            ap[l] = torch.nan
            continue

        # correct AP calculation
        # first append sentinel values at the end
        mpre = torch.cat((torch.tensor([0]), torch.nan_to_num(prec[l]), torch.tensor([0])))
        mrec = torch.cat((torch.tensor([0]), rec[l], torch.tensor([1])))
        
        # compute the precision envelope
        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap[l] = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, recalls
