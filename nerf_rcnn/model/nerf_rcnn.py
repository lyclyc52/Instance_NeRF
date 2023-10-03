"""
References:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py
"""

from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils import Matcher, BalancedPositiveNegativeSampler
from .utils import batched_box_iou, clip_boxes_to_mesh, remove_small_boxes, batched_nms, print_shape, roi_align_3d
from .utils import paste_masks_in_image
from .coder import MidpointOffsetCoder, AABBCoder
from .detector import ROIPool
from .rotated_iou.oriented_iou_loss import aabb2obb_3d
from .poolers import MultiScaleRoIAlign3D

import numpy as np
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.roi_heads import maskrcnn_inference


class NeRF_RCNN(nn.Module):
    """
    Implements Mask R-CNN over NeRF.

    The input to the model is expected to be a list of tensors, each of shape [C, W, L, H], one for each
    NeRF, and should be in 0-1 range. Different tensors can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (a list of dictionaries),
    each dictionary containing:
        - boxes (``FloatTensor[N, 7]``): the ground-truth boxes in ``[x, y, z, w, l, h, theta]`` format, with
          ``w >= 0, l >= 0, h >= 0, -pi/2 <= theta <= pi/2``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, W, L, H]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 7]``): the predicted boxes in ``[x, y, z, w, l, h, theta]`` format, with
          ``w >= 0, l >= 0, h >= 0, -pi/2 <= theta <= pi/2``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks (UInt8Tensor[N, 1, W, L, H]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        bbox_type (str): whether the boxes are AABBs or OBBs
        
        mask_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the mask head.
        mask_head (nn.Module): module that takes the cropped feature maps as input
        mask_predictor (nn.Module): module that takes the output of the mask_head and returns the
            segmentation mask logits
    """

    def __init__(
        self,
        # backbone,
        rpn_model,
        num_classes=None,
        use_input_rois=False,

        # Faster R-CNN parameters
        box_roi_pool=None,
        box_head=None,
        # box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        bbox_type='aabb',
        
        # Mask R-CNN parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        
        use_level_indices = False,
        **kwargs,
    ):
        
        super(NeRF_RCNN, self).__init__()
        
        self.nerf_rpn = rpn_model   
        self.use_input_rois = use_input_rois
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign3D, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign3D or None "
                f"instead of {type(box_roi_pool)}"
            )
        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign3D, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign3D or None "
                f"instead of {type(mask_roi_pool)}"
            )

        if num_classes is not None and mask_predictor is not None:
            raise ValueError("num_classes should be None when mask_predictor is specified")
        elif num_classes is None and mask_predictor is None:
            raise ValueError("num_classes should not be None when mask_predictor is not specified")

        out_channels = self.nerf_rpn.backbone.out_channels

        # Faster R-CNN head components
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign3D(output_size=5, sampling_ratio=-1)

        if box_head is None:
            resolution = int(box_roi_pool.output_size[0])
            rep_size = 512
            box_head = FastRCNNHead(out_channels * (resolution ** 3), rep_size, num_classes, bbox_type)

        # Mask R-CNN head components
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign3D(output_size=10, sampling_ratio=-1)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHead(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            # box_predictor,
            # Faster R-CNN training
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            bbox_type=bbox_type,
            # Faster R-CNN inference
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
            # Mask
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            # Others
            use_level_indices=use_level_indices
        )


    def forward(self, meshes=None, targets=None, rois=None):
        """
        Args:
            meshes (list[Tensor]): (Optional) feature grids to be processed.
            targets: (Optional) the targets for the meshes, represented as a list of dicts.
            containing: 
                - boxes (``FloatTensor[N, 7]``): the ground-truth boxes in ``[x, y, z, w, l, h, theta]`` format, with
                    ``w >= 0, l >= 0, h >= 0, -pi/2 <= theta <= pi/2``.
                - labels (Int64Tensor[N]): the class label for each ground-truth box
                - masks (UInt8Tensor[N, W, L, H]): the segmentation binary masks for each instance
            rois (List[proposals, level_indices]): (Optional)

        Returns:
            detections (List[Dict[Tensor]]): a list of dicts, one for each mesh, containing "boxes", "labels", "scores"
            losses (Dict[Tensor]) During training, it returns the losses for the RPN, Faster R-CNN and Mask R-CNN, 
                including "loss_objectness", "loss_rpn_box_reg", "loss_rpn_box_reg_2d", "loss_classifier", "loss_box_reg", 
                "loss_mask". During testing, it returns an empty dict.

        """
        mesh_shapes = [mesh.shape[1:] for mesh in meshes]

        # NeRF-RPN: get proposals
        # RPN porposals can only be used in inference after params are shared

        if not self.use_input_rois:
            assert not self.training, "RPN proposals can only be used in inference after params are shared"
            [features, proposals, level_index], proposal_losses, proposal_scores = self.nerf_rpn(meshes)
            proposals_pair = [proposals, level_index]
        else:
            # Transformation (0-padding) for batch size > 1
            meshes = self.nerf_rpn.transform(meshes)
            
            # Here we assume either the batch size is 1 or the grids are of the same size
            mesh_tensors = torch.stack(meshes, dim=0)
            features = list(self.nerf_rpn.backbone(mesh_tensors))
            proposals_pair = rois

        # roi_heads
        detections, detector_losses = self.roi_heads(features, proposals_pair, mesh_shapes, targets)
        
        losses = {}
        if self.training:
            # losses.update(proposal_losses)
            losses.update(detector_losses)

        return detections, losses


class FastRCNNHead(nn.Module):
    """
    Standard Two-layer MLP head with prediction layers

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, rep_size, num_classes, bbox_type):
        super().__init__()

        self.num_classes = num_classes
        if bbox_type == 'aabb':
            self.num_box_digits = 6
        elif bbox_type == 'obb':
            self.num_box_digits = 8
        else:
            raise ValueError("bbox_type should be 'aabb' or 'obb', got {}".format(bbox_type))
        
        self.fc6 = nn.Linear(in_channels, rep_size)
        self.fc7 = nn.Linear(rep_size, rep_size)
        self.cls_score = nn.Linear(rep_size, num_classes)
        self.bbox_pred = nn.Linear(rep_size, num_classes * self.num_box_digits)

    def forward(self, x_list):
        """
        Args:
            x_list (List[Tensor[B, C, W, L, H]])
        Returns:
            score_list (List[Tensor[B, N]])
            bbox_pred (List[Tensor[B, N, 6]] or List[Tensor[B, N, 8]])
        """
        score_list, delta_list = [], []
        for x in x_list:
            x = x.flatten(start_dim=1)
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            scores = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)
            score_list.append(scores.view(-1, self.num_classes))
            delta_list.append(bbox_deltas.view(-1, self.num_classes, self.num_box_digits))
        return score_list, delta_list


class MaskRCNNHead(nn.Sequential):
    _version = 2

    def __init__(self, in_channels, layers, dilation, norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv3dNormActivation(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_feature = layer_features

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self)
            for i in range(num_blocks):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}mask_fcn{i+1}.{type}"
                    new_key = f"{prefix}{i}.0.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose3d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv3d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class RoIHeads(nn.Module):

    def __init__(
        self,
        box_roi_pool,
        box_head,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        bbox_type,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # Others
        use_level_indices=False,
    ):
        super().__init__()

        self.box_similarity = batched_box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        # if bbox_reg_weights is None:
        #     bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        # self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_coder = MidpointOffsetCoder() if bbox_type == 'obb' else AABBCoder()
        self.box_pred_dim = 8 if bbox_type == 'obb' else 6

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        # self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        
        self.use_level_indices = use_level_indices

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
        self,
        proposals_pair,  # type: List[List[Tensor]]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):

        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        
        proposals = proposals_pair[0]
        proposal_indices = proposals_pair[1]
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        if not self.use_level_indices:
            proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Convert aabb proposals to obb representation.  
        # if len(proposals)>=1 and proposals[0].shape[1] == 6:
        #     proposals_obb = [aabb2obb_3d(p) for p in proposals]
        # else:
        #     proposals_obb = proposals


        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]

            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # proposals_obb[img_id] = proposals_obb[img_id][img_sampled_inds]
            if self.use_level_indices:
                proposal_indices[img_id] = proposal_indices[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, self.box_pred_dim), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals) # use aabb proposals to calculate regression targets
        return [proposals, proposal_indices], matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # TODO: decoder for multi-class cases
        pred_boxes = self.box_coder.decode(box_regression, proposals)   # (N, n_classes * box_dim)
        pred_boxes = pred_boxes.view(pred_boxes.size(0), -1, self.box_pred_dim)

        pred_scores = F.softmax(class_logits, -1)
        
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = clip_boxes_to_mesh(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, self.box_pred_dim)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  #type: List[List[Tensor]]
        proposals_pair,  # type: List[List[Tensor]]   
        meshes_shapes,  # type: List[Tuple[int, int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[List[Tensor])
            proposals_pair (It contains two lists of tensors, which are proposals and indices.)
            image_shapes (List[Tuple[W, L, H]])
            targets (List[Dict])

        Returns:
            result (List[Dict[Tensor]]): a list of dicts, one for each mesh, containing 'boxes', 'labels', 'scores'
            losses (Dict[Tensor]): a loss dict containing 'loss_classifier', 'loss_box_reg', 'loss_mask'
        """
        # check targets validity
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
        
        # Select proposals.
        if self.training:
            proposals_pair, matched_idxs, labels, regression_targets = self.select_training_samples(proposals_pair, targets)
        else:
            # proposals_pair[0] = [aabb2obb_3d(p) for p in proposals_pair[0]]
            matched_idxs = None
            labels = None
            regression_targets = None

        # Concatenate level indices in front of proposals
        proposals = proposals_pair[0] # List(Tensor[N, 6])
        roi_pooling_proposals = proposals
        
        # roi_pooling_proposals = []
        # for bbox, index in zip(proposals_pair[0], proposals_pair[1]):
        #     roi_pooling_proposals.append(torch.cat((index[:, None], bbox), dim=1))
            
        # roi_pooling_features = []
        # for batch in range(features[0].size(0)):
        #     batch_feature = []
        #     for level in features:
        #         batch_feature.append(level[batch])
        #     roi_pooling_features.append(batch_feature)

        # Detection head

        box_features = self.box_roi_pool(features, roi_pooling_proposals, 
                                         meshes_shapes) # box_features List[Tensor[B, C, W, L, H]]
        

        class_logits, box_regression = self.box_head(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            class_logits = torch.cat(class_logits, dim=0)
            box_regression = torch.cat(box_regression, dim=0)
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, meshes_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        # mask head
        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                # TODO: replace low-res feature roipool with high-res mesh roialign
                mask_features = self.mask_roi_pool(features, mask_proposals, meshes_shapes)
                cat_features = torch.cat(mask_features, dim=0)
                cat_features = self.mask_head(cat_features)
                mask_logits = self.mask_predictor(cat_features)
                # mask_logits = []
                # start = 0
                # for b in mask_features:
                #     mask_logits.append(cat_logits[start: start + b.size(0)])
                #     start += b.size(0)  
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r, proposal, shape in zip(masks_probs, result, mask_proposals, meshes_shapes):
                    mask_prob = mask_prob.squeeze(1)
                    r["masks"] = paste_masks_in_image(mask_prob, proposal, shape, 0.5)

            losses.update(loss_mask)

        return result, losses
    
    
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (List[Tensor])
        box_regression (List[Tensor])
        labels (list[Tensor])
        regression_targets (List[Tensor])

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    class_logits = torch.cat(class_logits, dim=0)
    box_regression = torch.cat(box_regression, dim=0)
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    # N, num_classes = class_logits.shape
    # box_regression = box_regression.reshape(N, num_classes, box_regression.size(-1) // num_classes)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align_3d(gt_masks, rois, (M, M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


if __name__ == '__main__':
    
    # testing code for rcnn model
    pass