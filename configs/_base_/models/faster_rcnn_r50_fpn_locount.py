# model settings
model = dict(
    type='FasterRCNNWithCount',
    backbone=dict(
        type='ResNet',
        depth=50,                                       # total layers (= 18, 34, 50, 101, 152).
        num_stages=4,                                   # total stages (<= 4).
        out_indices=(0, 1, 2, 3),                       # stage idxs that outputs (tuple) come from.
        frozen_stages=1,                                # stage **num** that no grad & eval mode.
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],             # num_ins = len(in_channels)
        out_channels=256,                               # Must be int! Same channels for all scales (outputs).
        num_outs=5),                                    # num of outputs (tuple)
    rpn_head=dict(
        type='RPNHead',                                 # RPN of 2-stage detector: binary cls, BG or Object. (num_class=1)
        in_channels=256,                                #                                /-> reg_conv(feat_chns, base_anchors*4)
        feat_channels=256,                              # N*rpn_conv(in_chns, feat_chns) --> cls_conv(feat_chns, base_anchors*num_classes)
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],                                 # global scale for every level.
            ratios=[0.5, 1.0, 2.0],                     # h/w for anchors of one point (base anchors). Same in all levels.
            strides=[4, 8, 16, 32, 64]),                # num_levels == len(strides), num_base_anchors == len(scales) * len(ratios).
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',                  # if ' means gt, we have:
            target_means=[.0, .0, .0, .0],              # deltas = [(x'-x)/w, (y'-y)/h, log(w'/w), log(h'/h)]
            target_stds=[1.0, 1.0, 1.0, 1.0]),          # deltas = (deltas - means) / stds
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadWithCount',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign',
                           output_size=7,               # output is a square, area = `output_size`**2
                           sampling_ratio=0),
            out_channels=256,                           # feature map's channels, from `neck` output.
            featmap_strides=[4, 8, 16, 32]),            # for multi-levels.
        bbox_head=dict(
            type='FCBBoxHeadWithCount',
            num_shared_fcs=2,
            in_channels=256,                            # from `bbox_roi_extractor` output.
            fc_out_channels=1024,                       # for every fc.
            roi_feat_size=7,                            # fc(cls/reg)'s in_channels = roi's area * `in_channels`. It's **fixed**.
            num_classes=140,
            num_counts=56,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_count_strategy=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0),
            loss_cnt=dict(
                type='CrossEntropyLoss',
                loss_weight=0.1))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,                        # > pos_iou_thr, pos samples.
                neg_iou_thr=0.3,                        # < neg_iou_thr, neg samples. In middle, excluded.
                min_pos_iou=0.3,                        # > min_pos_iou, match the RoI with gt. (when match_low_quality=True)
                match_low_quality=True,                 # Default.
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,                                # num after sampling.
                pos_fraction=0.5,                       # ratio of pos samples.
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerWithCount',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSamplerWithCount',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    ))
