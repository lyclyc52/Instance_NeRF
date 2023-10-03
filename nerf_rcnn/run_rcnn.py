import os
import glob
import json
import torch
import numpy as np
import argparse
import logging
import importlib.util
from copy import deepcopy

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.nerf_rpn import NeRFRegionProposalNetwork
from model.fcos.fcos import FCOSOverNeRF
from model.feature_extractor import Bottleneck, ResNet_FPN_256, ResNet_FPN_64, ResNetSimplified_64, ResNetSimplified_256
from model.feature_extractor import VGG_FPN, SwinTransformer_FPN
from model.anchor import AnchorGenerator3D, RPNHead
from model.nerf_rcnn import NeRF_RCNN
from model.utils import box_iou_3d, print_shape
from datasets import SegmentationDataset, Front3DSegmentationDataset
from eval import evaluate_map_recall
from model.poolers import MultiScaleRoIAlign3D
from model.nerf_rcnn import MaskRCNNHead, MaskRCNNPredictor, FastRCNNHead

from tqdm import tqdm
import wandb
import psutil


# Anchor parameters
anchor_sizes = ((8,), (16,), (32,), (64,),)
aspect_ratios = (((1., 1., 1.), (1., 1., 2.), (1., 2., 2.), (1., 1., 3.), 
                  (1., 3., 3.)),) * len(anchor_sizes)
normalize_aspect_ratios = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train and eval the NeRF-RCNN model.')

    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'benchmark'])

    # data path
    parser.add_argument('--dataset_root', '-dr', default='', help='The path to the dataset root.')

    parser.add_argument('--save_path', default='', help='The path to save the model.')
    parser.add_argument('--dataset_split', default=None, help='The dataset split to use.')
    parser.add_argument('--rcnn_ckpt', default='', help='The path to the RCNN checkpoint.')
    parser.add_argument('--rpn_ckpt', default='', help='The path to the RPN checkpoint.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')

    parser.add_argument('--backbone_type', type=str, default='resnet', 
                        choices=['resnet', 'vgg_AF', 'vgg_EF',  'swin_t', 'swin_s', 'swin_b', 'swin_l'],)
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the backbone.')
    parser.add_argument('--rpn_type', choices=['anchor', 'fcos'], default='anchor')

    parser.add_argument('--resolution', type=int, default=256, help='The max resolution of the input features.')
    parser.add_argument('--bbox_type', default='aabb', choices=['obb', 'aabb'],
                        help='obb: [x, y, z, w, h, d, theta] \
                              If no, bbox: (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]')
    parser.add_argument('--normalize_density', action='store_true', help='Whether to normalize the density.')

    parser.add_argument('--filter', choices=['none', 'tp', 'fp'], default='none', 
                        help='Filter for the proposal output.')
    parser.add_argument('--filter_threshold', type=float, default=0.7,
                        help='The IoU threshold for the proposal filter, only used if --output_proposals is True '
                        'and --filter is not "none".')
    parser.add_argument('--top_k', type=int, default=None,
                        help='The number of proposals that will be used to calculate AP.')
    parser.add_argument('--save_top_k', type=int, default=30,
                        help='The number of proposals whose 3D masks will be saved.')
    
    parser.add_argument('--rotate_prob', default=0.5, type=float, help='The probability of rotating the scene.')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='The probability of flipping the scene.')
    parser.add_argument('--rot_scale_prob', default=0.5, type=float, help='The probability of extra rotation and scaling.')

    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help='The batch size.')
    parser.add_argument('--num_epochs', default=100, type=int, help='The number of epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate.')
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='The weight decay coefficient of AdamW.')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float, help='The gradient clipping norm.')

    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    parser.add_argument('--log_interval', default=20, type=int, help='The number of iterations to print the loss.')
    parser.add_argument('--eval_interval', default=1, type=int, help='The number of epochs to evaluate.')
    parser.add_argument('--keep_checkpoints', default=5, type=int, help='The number of latest checkpoints to keep.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')


    # Distributed training parameters
    parser.add_argument('--gpus', default='', help='The gpus to use for distributed training. If empty, '
                        'uses the first available gpu. DDP is only enabled if this is greater than one.')

    # RPN model parameters
    parser.add_argument('--rpn_head_conv_depth', default=4, type=int, 
                        help='The number of common convolutional layers in the RPN head.')
    parser.add_argument('--rpn_pre_nms_top_n_train', default=2500, type=int, 
                        help='The number of top proposals to keep before applying NMS.')
    parser.add_argument('--rpn_pre_nms_top_n_test', default=1000, type=int,
                        help='The number of top proposals to keep before applying NMS.')
    parser.add_argument('--rpn_post_nms_top_n_train', default=2500, type=int,
                        help='The number of top proposals to keep after applying NMS.')
    parser.add_argument('--rpn_post_nms_top_n_test', default=1000, type=int,
                        help='The number of top proposals to keep after applying NMS.')
    parser.add_argument('--rpn_nms_thresh', default=0.5, type=float,
                        help='The NMS threshold.')
    parser.add_argument('--rpn_fg_iou_thresh', default=0.35, type=float,
                        help='The foreground IoU threshold.')
    parser.add_argument('--rpn_bg_iou_thresh', default=0.2, type=float,
                        help='The background IoU threshold.')
    parser.add_argument('--rpn_batch_size_per_mesh', default=256, type=int,
                        help='The batch size per mesh.')
    parser.add_argument('--rpn_positive_fraction', default=0.5, type=float,
                        help='The fraction of positive proposals to use.')
    parser.add_argument('--rpn_score_thresh', default=0.0, type=float,
                        help='The score threshold.')
    parser.add_argument('--reg_loss_type', choices=['smooth_l1', 'iou', 'linear_iou', 'giou', 'diou'], 
                        default='smooth_l1', help='The type of regression loss to use for the RPN.')
    
    # RCNN parameters. Detailed description can be found in ./model/nerf_rcnn.py:NeRF_RCNN
    parser.add_argument('--box_roi_pool_output_size', default=5, type=int,
                        help='The roi pooling output size.')
    parser.add_argument('--box_roi_pool_sample_ratio', default=-1, type=int,
                        help='The roi pooling sampling size. Refer to model.utils.py:roi_align_3d for more details')
    parser.add_argument('--mask_roi_pool_output_size', default=10, type=int,
                        help='The roi pooling output size.')
    parser.add_argument('--mask_roi_pool_sample_ratio', default=-1, type=int,
                        help='The roi pooling sampling size. Refer to model.utils.py:roi_align_3d for more details')
    parser.add_argument('--mask_head_layer', nargs='+', type=int, default=[256,256,256,256],
                        help='Number of layers and output dimensions in mask head')
    parser.add_argument('--mask_head_dilation', default=1, type=int,
                        help='Dilation in mask head.')
    parser.add_argument('--use_level_indices', action='store_true', 
                        help='Whether to use the level indices from FPN. ROI align will only use the feature from one level if it is true.')

    parser.add_argument('--RCNN_box_score_thresh', default=0.01, type=float, 
                        help='The box score threshold in the RCNN head.')
    parser.add_argument('--RCNN_box_nms_thresh', default=0.2, type=float, 
                        help='The box NMS threshold in the RCNN head.')   
    parser.add_argument('--RCNN_box_detections_per_img', default=100, type=int, 
                        help='The maximum number of detected boxes per image.')  
    parser.add_argument('--RCNN_box_fg_iou_thresh', default=0.25, type=float, 
                        help='The min IOU threshold for foreground/positive objects.')  
    parser.add_argument('--RCNN_box_bg_iou_thresh', default=0.25, type=float, 
                        help='The max IOU threshold for background/negative objects.')  
    parser.add_argument('--RCNN_box_batch_size_per_image', default=512, type=int, 
                        help='Batch size for RCNN model.')  
    parser.add_argument('--RCNN_box_positive_fraction', default=0.25, type=float, 
                        help='Proportion of positive samples per batch.') 
    parser.add_argument('--RCNN_bbox_reg_weights', nargs='+', type=float, default=None, 
                        help='')                              
    parser.add_argument('--use_input_rois', action='store_true',
                        help='Use pre-computed rois as input if it is true. Otherwise, '
                             'use RPN model to compute online')
    
    
    parser.add_argument('--check_arch', action='store_true', 
                        help='Check the model architecture, then exit.')
    parser.add_argument('--max_cpu_cores', default=0, type=int,
                        help='The maximum number of CPU cores to use.')

    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args, rank=0, world_size=1, device_id=None, logger=None):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self.logger = logger if logger is not None else logging.getLogger()

        # Limiting the number of CPU cores to use
        if args.max_cpu_cores > 0:
            cur_process = psutil.Process()
            cpu_affinity = cur_process.cpu_affinity()
            cpu_affinity = sorted(cpu_affinity)[:args.max_cpu_cores]
            cur_process.cpu_affinity(cpu_affinity)
            if rank == 0:
                self.logger.info(f'Limited to {args.max_cpu_cores} CPU cores')

        if args.wandb and rank == 0:
            project_name = f'nerf_rcnn'
            wandb.init(project=project_name)
            wandb.config = deepcopy(args)

        self.logger.info('Constructing model.')
        
        self.num_bbox_digits = 6 if args.bbox_type=='aabb' else 7

        # Only support 3D-FRONT dataset for now
        self.n_classes = Front3DSegmentationDataset.NUM_CLASSES + 1 # +1 for background

        self.logger.info(f'Number of classes: {self.n_classes}')

        if args.bbox_type == 'obb':
            self.logger.error('OBB is not supported yet.')
            raise NotImplementedError

        self.build_rpn_model()

        self.build_rcnn_heads()

        self.build_rcnn()

        self.load_ckpt()

        if torch.cuda.is_available():
            self.model.cuda()

        if args.check_arch:
            print("Checking model architecture on GPU... (will exit after printing torchinfo summary)")
            spec = importlib.util.find_spec('torchinfo')
            self.model.train()
            if spec is not None:
                meshes = [torch.rand(4, 256, 256, 256).cuda()]
                gt_boxes = torch.rand(2, 7).cuda()
                gt_boxes[:, 3:6] += 1.0
                gt_boxes = [gt_boxes]
                gt_labels = [torch.randint(0, 2, (2,)).cuda()]
                gt_masks = [torch.randint(0, 2, (2, 256, 256, 256)).cuda()]
                targets = []
                for box, label, mask in zip(gt_boxes, gt_labels, gt_masks):
                    targets.append({"boxes": box, "labels": label, "masks": mask})
                # import torchinfo
                # torchinfo.summary(self.model, input_data=(meshes, targets))
                output = self.model(meshes, targets)
            else:
                self.logger.info(self.model)
            exit()

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device_id], find_unused_parameters=True)

        # if args.wandb and rank == 0:
        #     wandb.watch(self.model, log_freq=10)

        self.init_datasets()

    def init_datasets(self):
        # So far, we only tested on the 3D-FRONT dataset
        transpose_yz = False
        dataset = Front3DSegmentationDataset

        if self.args.mode == 'train':
            # Create dataset
            self.train_set = dataset(
                mode='train', 
                root_dir=self.args.dataset_root,
                data_split=self.args.dataset_split,
                normalize_density=self.args.normalize_density,
                transpose_yz=transpose_yz,
            )
            
            self.val_set = dataset(
                mode='val', 
                root_dir=self.args.dataset_root,
                data_split=self.args.dataset_split,
                normalize_density=self.args.normalize_density,
                transpose_yz=transpose_yz,
            )

            if self.rank == 0:
                self.logger.info(f'Loaded {len(self.train_set)} training scenes')
                self.logger.info(f'Loaded {len(self.val_set)} validation scenes')

        if self.args.mode == 'eval':
            self.test_set = dataset(
                mode='test', 
                root_dir=self.args.dataset_root,
                data_split=self.args.dataset_split,
                normalize_density=self.args.normalize_density,
                transpose_yz=transpose_yz,
            )

            if self.rank == 0:
                self.logger.info(f'Loaded {len(self.test_set)} test scenes')

    def build_rcnn(self):
        self.model = NeRF_RCNN(
            rpn_model=self.rpn_model,
            num_classes=None if self.mask_predictor else self.n_classes,
            use_input_rois=self.args.use_input_rois,
            
            # Faster R-CNN parameters
            box_roi_pool = self.box_roi_pool,
            box_head = self.box_head,

            box_score_thresh=self.args.RCNN_box_score_thresh,
            box_nms_thresh=self.args.RCNN_box_nms_thresh,
            box_detections_per_img=self.args.RCNN_box_detections_per_img,
            box_fg_iou_thresh=self.args.RCNN_box_fg_iou_thresh,
            box_bg_iou_thresh=self.args.RCNN_box_bg_iou_thresh,
            box_batch_size_per_image=self.args.RCNN_box_batch_size_per_image,
            box_positive_fraction=self.args.RCNN_box_positive_fraction,
            bbox_reg_weights=self.args.RCNN_bbox_reg_weights,
            bbox_type=self.args.bbox_type,
        
            # Mask R-CNN parameters
            mask_roi_pool = self.mask_roi_pool,
            mask_head = self.mask_head,
            mask_predictor = self.mask_predictor,
            
            use_level_indices=self.args.use_level_indices
        )

    def build_rcnn_heads(self):
        self.box_roi_pool = MultiScaleRoIAlign3D(output_size=self.args.box_roi_pool_output_size, 
                                                 sampling_ratio=self.args.box_roi_pool_sample_ratio)
        resolution = int(self.box_roi_pool.output_size[0])
        rep_size = 512
        self.box_head = FastRCNNHead(self.backbone.out_channels * (resolution ** 3), rep_size, self.n_classes, self.args.bbox_type)
        
        self.mask_roi_pool = MultiScaleRoIAlign3D(output_size=self.args.mask_roi_pool_output_size, 
                                                  sampling_ratio=self.args.mask_roi_pool_sample_ratio)
        
        mask_layers = self.args.mask_head_layer
        self.mask_head = MaskRCNNHead(self.backbone.out_channels, mask_layers, self.args.mask_head_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, self.n_classes)

    def build_rpn_model(self):
        '''
        This function will create a NeRF RPN model using `args` 
        '''
        self.build_backbone()

        if self.args.rpn_type == 'anchor':
            self.anchor_generator = AnchorGenerator3D(anchor_sizes, aspect_ratios, 
                                                    is_normalized=normalize_aspect_ratios)

            # Assuming the number of anchors are the same for all levels of features
            self.rpn_head = RPNHead(self.backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0],
                                    self.args.rpn_head_conv_depth, rotate=True)   # allows to use RPN checkpoint with yaw

            if self.args.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
            self.rpn_model = NeRFRegionProposalNetwork(
                self.backbone, 
                self.anchor_generator, 
                self.rpn_head,
                rpn_pre_nms_top_n_train=self.args.rpn_pre_nms_top_n_train,
                rpn_pre_nms_top_n_test=self.args.rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_train=self.args.rpn_post_nms_top_n_train,
                rpn_post_nms_top_n_test=self.args.rpn_post_nms_top_n_test,
                rpn_nms_thresh=self.args.rpn_nms_thresh,
                rpn_fg_iou_thresh=self.args.rpn_fg_iou_thresh,
                rpn_bg_iou_thresh=self.args.rpn_bg_iou_thresh,
                rpn_batch_size_per_mesh=self.args.rpn_batch_size_per_mesh,
                rpn_positive_fraction=self.args.rpn_positive_fraction,
                rpn_score_thresh=self.args.rpn_score_thresh,
                res=self.args.resolution,
                rotated_bbox=True,    # allows to use RPN checkpoint with yaw
                reg_loss_type=self.args.reg_loss_type,
            )

        elif self.args.rpn_type == 'fcos':
            assert self.args.rpn_ckpt or self.args.rcnn_ckpt, 'Either rpn_ckpt or rcnn_ckpt must be provided.'
            if self.args.rcnn_ckpt:
                checkpoint = torch.load(self.args.rcnn_ckpt, map_location='cpu')
                train_args = checkpoint['fcos_train_args']
            else:
                checkpoint = torch.load(self.args.rpn_ckpt, map_location='cpu')
                train_args = checkpoint['train_args']

            train_args = argparse.Namespace(**train_args)

            # TODO: params override for inference
            self.rpn_model = FCOSOverNeRF(
                args=train_args,
                backbone=self.backbone,
                fpn_strides=[4, 8, 16, 32],
                world_size=self.world_size
            )

            self.rpn_head = self.rpn_model.fcos_module

    def load_ckpt(self):
        assert self.args.rpn_ckpt or self.args.rcnn_ckpt, 'Either rpn_ckpt or rcnn_ckpt must be provided.'

        if self.args.rcnn_ckpt:
            assert os.path.exists(self.args.rcnn_ckpt), f'{self.args.rcnn_ckpt} does not exist.'
            self.logger.info(f'Loading RCNN checkpoint from {self.args.rcnn_ckpt}.')
            checkpoint = torch.load(self.args.rcnn_ckpt, map_location='cpu')

            if self.rank == 0:
                print('Training args from RCNN checkpoint:')
                for k, v in checkpoint['train_args'].items():
                    print(f'{k}: {v}')

            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            self.rpn_head.load_state_dict(checkpoint['rpn_head_state_dict'])
            self.model.roi_heads.load_state_dict(checkpoint['roi_heads_state_dict'])
            for param in self.rpn_head.parameters():
                param.requires_grad = False

        # Load RPN checkpoint
        elif self.args.rpn_ckpt:
            assert os.path.exists(self.args.rpn_ckpt), f'{self.args.rpn_ckpt} does not exist.'
            self.logger.info(f'Loading RPN checkpoint from {self.args.rpn_ckpt}.')
            checkpoint = torch.load(self.args.rpn_ckpt, map_location='cpu')

            if self.rank == 0:
                print('Training args from RPN backbone:')
                for k, v in checkpoint['train_args'].items():
                    print(f'{k}: {v}')

            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            if self.args.rpn_type == 'anchor':
                self.rpn_head.load_state_dict(checkpoint['rpn_head_state_dict'])
            elif self.args.rpn_type == 'fcos':
                self.rpn_head.load_state_dict(checkpoint['fcos_state_dict'])

            for param in self.rpn_head.parameters():
                param.requires_grad = False

    def build_backbone(self):
        if self.args.backbone_type == 'resnet':
            self.backbone = ResNet_FPN_256(Bottleneck, [3, 4, 6, 3], input_dim=4, is_max_pool=True)
        elif self.args.backbone_type == 'vgg_AF':
            self.backbone = VGG_FPN("AF", 4, True, self.args.resolution)
        elif self.args.backbone_type == 'vgg_EF':
            self.backbone = VGG_FPN("EF", 4, True, self.args.resolution)
        elif self.args.backbone_type.startswith('swin'):
            swin = {'swin_t': {'embed_dim':96, 'depths':[2, 2, 6, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_s': {'embed_dim':96, 'depths':[2, 2, 18, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_b': {'embed_dim':128, 'depths':[2, 2, 18, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_l': {'embed_dim':192, 'depths':[2, 2, 18, 2], 'num_heads':[6, 12, 24, 48]}}
            self.backbone = SwinTransformer_FPN(patch_size=[4, 4, 4], 
                                                embed_dim=swin[self.args.backbone_type]['embed_dim'], 
                                                depths=swin[self.args.backbone_type]['depths'],
                                                num_heads=swin[self.args.backbone_type]['num_heads'], 
                                                window_size=[4, 4, 4],
                                                stochastic_depth_prob=0.1,
                                                expand_dim=True)

    def save_checkpoint(self, epoch, path):
        if self.world_size == 1:
            model = self.model
        else:
            model = self.model.module
        
        roi_heads_state_dict = model.roi_heads.state_dict()
        fcos_train_args = model.nerf_rpn.args.__dict__ if self.args.rpn_type == 'fcos' else None

        torch.save({
            'epoch': epoch,

            'backbone_state_dict': self.backbone.state_dict(),
            'rpn_head_state_dict': self.rpn_head.state_dict(),
            'roi_heads_state_dict': roi_heads_state_dict,

            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_args': self.args.__dict__,
            'fcos_train_args': fcos_train_args
        }, path)

    def delete_old_checkpoints(self, path, keep_latest=5):
        files = glob.glob(f'{path}/epoch_*.pt')
        files.sort(key=os.path.getmtime)
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                logging.info(f'Deleting old checkpoint {file}.')
                os.remove(file)

    def train_loop(self):

        # Create DataLoader
        if self.world_size == 1:
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, 
                                           collate_fn=SegmentationDataset.collate_fn,
                                           shuffle=True, num_workers=4, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(self.train_set)
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size // self.world_size,
                                           collate_fn=SegmentationDataset.collate_fn,
                                           sampler=self.train_sampler, num_workers=0, pin_memory=False)

        # Create optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, 
                               weight_decay=self.args.weight_decay)

        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.args.lr,
                                    total_steps=self.args.num_epochs * len(self.train_loader))

        # Load optimizer and scheduler checkpoints
        start_epoch = 0
        if self.args.rcnn_ckpt and self.args.resume:
            checkpoint = torch.load(self.args.rcnn_ckpt, map_location='cpu')
            if 'optimizer_state_dict' in checkpoint.keys():
                self.logger.info(f'Loading optimizer from RCNN checkpoint')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # self.optimizer = self.optimizer.cuda()
            if 'scheduler_state_dict' in checkpoint.keys():
                self.logger.info(f'Loading scheduler from RCNN checkpoint')
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # self.scheduler = self.scheduler.cuda()
            start_epoch = checkpoint['epoch']

        self.best_metric = None
        os.makedirs(self.args.save_path, exist_ok=True)

        for epoch in range(start_epoch, self.args.num_epochs):
            if self.world_size > 1:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(epoch)
            if self.rank != 0:
                continue

            if (epoch % self.args.eval_interval == 0) or epoch == self.args.num_epochs - 1:
                results = self.eval(self.val_set)
                metric = results['mAP_25']
                if self.best_metric is None or metric > self.best_metric:
                    self.best_metric = metric
                    self.save_checkpoint(epoch, os.path.join(self.args.save_path, 'model_best.pt'))

                self.save_checkpoint(epoch, os.path.join(self.args.save_path, f'epoch_{epoch}.pt'))
                self.delete_old_checkpoints(self.args.save_path, keep_latest=self.args.keep_checkpoints)
   
    def train_epoch(self, epoch):
        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(self.train_loader):

            # self.logger.debug(f'GPU {self.device_id} Epoch {epoch} Iter {i} {batch[0]}')

            self.model.train()
            
            # For each scene, the input contains a RGB and sigma grid, 
            # 3D oriented bounding boxes and the segmentation with class label of each objects
            scenes, features, boxes, class_ids, masks, rois = batch
            
            if torch.cuda.is_available():
                feature = [item.cuda() for item in features]
                boxes = [item.cuda() for item in boxes]
                class_ids = [item.cuda() for item in class_ids]
                
                bboxes = [item[0].cuda() for item in rois]
                indices = [item[1].cuda() for item in rois]
                rois = [bboxes, indices]
                
            # The initial resolution is high so the grid has to be downsampled before being fed into 
            # RPN model
            targets = []
            for box, label, mask in zip(boxes, class_ids, masks):
                targets.append({'boxes': box, 'labels': label, 'masks': mask})
            
            detections, losses = self.model(meshes=feature, 
                                            rois=rois, targets=targets)
            
            # TODO: Weights?
            loss = losses['loss_classifier'] + losses['loss_box_reg'] + losses['loss_mask']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            
            self.optimizer.zero_grad()

            # self.logger.debug(f'GPU {self.device_id} Epoch {epoch} Iter {i} {batch[-1]} '
            #                   f'cls_loss: {losses["loss_classifier"].item():.6f} '
            #                   f'box_reg_loss: {losses["loss_box_reg"].item():.6f} '
            #                   f'mask_loss: {losses["loss_mask"].item():.6f}')

            if self.world_size > 1:
                dist.barrier()
                dist.all_reduce(loss)

                loss /= self.world_size

            if i % self.args.log_interval == 0 and self.rank == 0:
                self.logger.info(f'epoch {epoch} [{i}/{len(self.train_loader)}] '
                                 f'loss: {loss.item():.4f} '
                                 f'cls_loss: {losses["loss_classifier"].item():.4f} '
                                 f'box_reg_loss: {losses["loss_box_reg"].item():.4f} '
                                 f'mask_loss: {losses["loss_mask"].item():.4f}')

            if self.args.wandb and self.rank == 0:
                wandb.log({
                    'lr': self.scheduler.get_last_lr()[0],
                    'loss': loss.item(),
                    'cls_loss': losses['loss_classifier'].item(),
                    'box_reg_loss': losses['loss_box_reg'].item(),
                    'mask_loss': losses['loss_mask'].item(),
                    'epoch': epoch,
                    'iter': i,
                })


    @torch.no_grad()
    def eval(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size // self.world_size, 
                                shuffle=False, num_workers=0,
                                collate_fn=dataset.collate_fn)

        self.logger.info(f'Evaluating...')

        masks_list = []
        scores_list = []
        bboxes_list = []
        labels_list = []
        gt_masks_list = []
        gt_boxes_list = []
        gt_cls_list = []        
        scenes_list = []
        
        for batch in tqdm(dataloader):
            scenes, feature, gt_boxes, gt_class_ids, gt_masks, rois = batch
            if torch.cuda.is_available():
                feature = [item.cuda() for item in feature]
                
                roi_boxes = [item[0][:20].cuda() for item in rois]
                indices = [item[1][:20].cuda() for item in rois]
                rois = [roi_boxes, indices]

            output_list, losses  = self.model(meshes=feature, rois=rois)
            
            bboxes = [output['boxes'].cpu() for output in output_list]
            masks = [output['masks'].cpu() for output in output_list]
            scores = [output['scores'].cpu() for output in output_list]
            labels = [output['labels'].cpu() for output in output_list]
            
            scenes_list.extend(scenes)
            masks_list.extend(masks)
            scores_list.extend(scores)
            bboxes_list.extend(bboxes)
            labels_list.extend(labels)
            
            gt_masks_list.extend(gt_masks)
            gt_boxes_list.extend(gt_boxes)
            gt_cls_list.extend(gt_class_ids)
            
            torch.cuda.empty_cache()

        os.makedirs(os.path.join(self.args.save_path, 'masks'), exist_ok=True)
        for i in range(len(masks_list)):
            score = scores_list[i].numpy()
            label = labels_list[i].numpy()
            mask = masks_list[i].numpy()
            box = bboxes_list[i].numpy()

            top_k = min(self.args.save_top_k, len(score))
            inds = np.argsort(score)[::-1][:top_k]
            score = score[inds]
            label = label[inds]
            mask = mask[inds]
            box = box[inds]
            
            np.savez(os.path.join(self.args.save_path, 'masks', f'{scenes_list[i]}.npz'), 
                     masks=mask, scores=score, labels=label, boxes=box)

        if gt_masks_list[0] is None:
            return None     # no ground truth

        # Average precisions
        ap50, recall50 = evaluate_map_recall(masks_list, scores_list, labels_list, gt_masks_list, gt_cls_list,
                                             iou_thresh=0.5, top_k=self.args.top_k, iou_type='mask')
        ap25, recall25 = evaluate_map_recall(masks_list, scores_list, labels_list, gt_masks_list, gt_cls_list,
                                             iou_thresh=0.25, top_k=self.args.top_k, iou_type='mask')

        mAP50 = torch.mean(ap50[~torch.isnan(ap50)])
        mAP25 = torch.mean(ap25[~torch.isnan(ap25)])
        AR50 = torch.mean(recall50[~torch.isnan(recall50)])
        AR25 = torch.mean(recall25[~torch.isnan(recall25)])

        print(f'ap50: {ap50}')
        print(f'ap25: {ap25}')
        print(f'recall50: {recall50}')
        print(f'recall25: {recall25}')

        print(f'mAP_50: {mAP50.item():.4f}')
        print(f'mAP_25: {mAP25.item():.4f}')
        print(f'AR_50: {AR50.item():.4f}')
        print(f'AR_25: {AR25.item():.4f}')

        box_ap50, box_recall50 = evaluate_map_recall(bboxes_list, scores_list, labels_list, gt_boxes_list, gt_cls_list,
                                                     iou_thresh=0.5, top_k=self.args.top_k, iou_type='box')
        box_ap25, box_recall25 = evaluate_map_recall(bboxes_list, scores_list, labels_list, gt_boxes_list, gt_cls_list,
                                                     iou_thresh=0.25, top_k=self.args.top_k, iou_type='box')

        box_mAP50 = torch.mean(box_ap50[~torch.isnan(box_ap50)])
        box_mAP25 = torch.mean(box_ap25[~torch.isnan(box_ap25)])
        box_AR50 = torch.mean(box_recall50[~torch.isnan(box_recall50)])
        box_AR25 = torch.mean(box_recall25[~torch.isnan(box_recall25)])

        print(f'box_ap50: {box_ap50}')
        print(f'box_ap25: {box_ap25}')
        print(f'box_recall50: {box_recall50}')
        print(f'box_recall25: {box_recall25}')

        print(f'box_mAP_50: {box_mAP50.item():.4f}')
        print(f'box_mAP_25: {box_mAP25.item():.4f}')
        print(f'box_AR_50: {box_AR50.item():.4f}')
        print(f'box_AR_25: {box_AR25.item():.4f}')

        results = {
            'mAP_50': mAP50.item(),
            'mAP_25': mAP25.item(),
            'AR_50': AR50.item(),
            'AR_25': AR25.item(),
            'box_mAP_50': box_mAP50.item(),
            'box_mAP_25': box_mAP25.item(),
            'box_AR_50': box_AR50.item(),
            'box_AR_25': box_AR25.item(),
        }

        if self.args.wandb:
            wandb.log(results, commit=True)

        return results

    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    @torch.no_grad()
    def benchmark(self):
        dummy_input = [torch.randn(4, 200, 200, 130, dtype=torch.float).cuda()]
        self.model.eval()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = self.model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f'Average inference time: {mean_syn:.4f} ms, std: {std_syn:.4f} ms')


def main_worker(proc, nprocs, args, gpu_ids, init_method):
    '''
    Main worker function for multiprocessing.
    '''
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=nprocs, rank=proc)
    torch.cuda.set_device(gpu_ids[proc])

    logger = logging.getLogger(f'worker_{proc}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.log_to_file:
        log_dir = os.path.join(args.save_path, 'log')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f'worker_{proc}.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    trainer = Trainer(args, proc, nprocs, gpu_ids[proc], logger)
    dist.barrier()
    if args.mode == 'train':
        trainer.train_loop()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')

    gpu_ids = []
    if args.gpus:
        for token in args.gpus.split(','):
            if '-' in token:
                start, end = token.split('-')
                gpu_ids.extend(range(int(start), int(end)+1))
            else:
                gpu_ids.append(int(token))

    if len(gpu_ids) <= 1:
        if len(gpu_ids) == 1:
            torch.cuda.set_device(gpu_ids[0])

        logger = None
        if args.log_to_file:
            log_dir = os.path.join(args.save_path, 'log')
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f'worker_0.log'))
            file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
            file_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger('worker_0')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        trainer = Trainer(args, logger=logger)

        if args.mode == 'train':
            trainer.train_loop()
        elif args.mode == 'eval':
            trainer.eval(trainer.test_set)
        elif args.mode == 'benchmark':
            trainer.benchmark()
    else:
        init_method = f'tcp://127.0.0.1:{np.random.randint(20000, 40000)}'
        nprocs = len(gpu_ids)
        logging.info(f'Using {nprocs} processes for DDP, GPUs: {gpu_ids}')
        mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, args, gpu_ids, init_method), join=True)


if __name__ == '__main__':
    main()
