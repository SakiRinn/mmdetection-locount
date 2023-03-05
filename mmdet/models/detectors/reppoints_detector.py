# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector, SingleStageDetectorWithCount


@DETECTORS.register_module()
class RepPointsDetector(SingleStageDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)


@DETECTORS.register_module()
class RepPointsDetectorWithCount(SingleStageDetectorWithCount):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RepPointsDetectorWithCount,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)