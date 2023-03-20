# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector, SingleStageDetectorWithCount


@DETECTORS.register_module()
class PAA(SingleStageDetector):
    """Implementation of `PAA <https://arxiv.org/pdf/2007.08103.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PAA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)


@DETECTORS.register_module()
class PAAWithCount(SingleStageDetectorWithCount):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PAAWithCount, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained, init_cfg)