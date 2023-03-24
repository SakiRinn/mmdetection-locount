# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector, SingleStageDetectorWithCount


@DETECTORS.register_module()
class DDOD(SingleStageDetector):
    """Implementation of `DDOD <https://arxiv.org/pdf/2107.02963.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DDOD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)


@DETECTORS.register_module()
class DDODWithCount(SingleStageDetectorWithCount):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DDODWithCount, self).__init__(backbone, neck, bbox_head, train_cfg,
                                            test_cfg, pretrained, init_cfg)
