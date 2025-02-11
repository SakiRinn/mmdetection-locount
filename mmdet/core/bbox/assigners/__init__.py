# Copyright (c) OpenMMLab. All rights reserved.
from .approx_max_iou_assigner import ApproxMaxIoUAssigner
from .assign_result import AssignResult, AssignResultWithCount
from .atss_assigner import ATSSAssigner
from .base_assigner import BaseAssigner, BaseAssignerWithCount
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .hungarian_assigner import HungarianAssigner, HungarianAssignerWithCount
from .mask_hungarian_assigner import MaskHungarianAssigner
from .max_iou_assigner import MaxIoUAssigner, MaxIoUAssignerWithCount
from .point_assigner import PointAssigner, PointAssignerWithCount
from .region_assigner import RegionAssigner
from .sim_ota_assigner import SimOTAAssigner
from .task_aligned_assigner import TaskAlignedAssigner
from .uniform_assigner import UniformAssigner, UniformAssignerWithCount

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner', 'RegionAssigner', 'UniformAssigner', 'SimOTAAssigner',
    'TaskAlignedAssigner', 'MaskHungarianAssigner',
    'AssignResultWithCount', 'BaseAssignerWithCount',
    'MaxIoUAssignerWithCount', 'PointAssignerWithCount', 'HungarianAssignerWithCount',
    'UniformAssignerWithCount'
]
