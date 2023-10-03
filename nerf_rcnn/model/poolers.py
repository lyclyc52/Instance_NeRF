from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from .utils import roi_align_3d, print_shape


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 160,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


def box_volume(boxes: Tensor) -> Tensor:
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 160,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.pow(torch.cat([box_volume(boxlist) for boxlist in boxlists]), 1.0 / 3.0)

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-3:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    return possible_scales[0]


def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    if not image_shapes:
        raise ValueError("images list should not be empty")
    max_x = 0
    max_y = 0
    max_z = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
        max_z = max(shape[2], max_z)
    original_input_shape = (max_x, max_y, max_z)

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


def _multiscale_roi_align_3d(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: List[int],
    sampling_ratio: int,
    scales: List[float],
    mapper: LevelMapper,
) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 6]]): boxes to be used to perform the pooling operation, in
            (x1, y1, z1, x2, y2, z2) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2``, ``0 <= y1 < y2``, and
            ``0 <= z1 < z2``.
        output_size (Union[List[Tuple[int, int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (List[float]): spatial scales for each input feature map
        mapper (LevelMapper)
    Returns:
        result (Tensor)
    """
    if scales is None or mapper is None:
        raise ValueError("scales and mapper should not be None")

    num_levels = len(x_filtered)
    rois = _convert_to_roi_format(boxes)

    if num_levels == 1:
        return roi_align_3d(
            x_filtered[0],
            rois,
            output_size=output_size,
            spatial_scale=scales[0],
            sampling_ratio=sampling_ratio,
        )

    levels = mapper(boxes)
    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = torch.zeros(
        (
            num_rois,
            num_channels,
        )
        + output_size,
        dtype=dtype,
        device=device,
    )

    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        if idx_in_level.size(0) == 0:
            continue
        
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align_3d(
            per_level_feature,
            rois_per_level,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )

        result[idx_in_level] = result_idx_in_level.to(result.dtype)
    
    output = []
    for b in range(len(boxes)):
        index = torch.where(rois[:, 0] == b)
        output.append(result[index])
    return output


class MultiScaleRoIAlign3D(nn.Module):
    """
    Multi-scale 3D RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.

    Args:
        output_size (List[Tuple[int, int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 160,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: List[Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int, int]],
    ) -> Tensor:
        """
        Args:
            x (List[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 6]]): boxes to be used to perform the pooling operation, in
                (x1, y1, z1, x2, y2, z2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2``, ``0 <= y1 < y2``,
                and ``0 <= z1 < z2``.
            image_shapes (List[Tuple[width, length, height]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x, image_shapes, self.canonical_scale, self.canonical_level
            )
        return _multiscale_roi_align_3d(
            x,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
