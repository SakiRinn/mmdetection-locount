# model settings
model = dict(
    type='RetinaNetWithCount',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,                                  # faster: None.
        add_extra_convs='on_input',                     # faster: None.
        num_outs=5),
    bbox_head=dict(
        type='RetinaHeadWithCount',
        num_classes=140,
        num_counts=56,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(                          # faeter: in `rpn_head`, add `scales=[8]`.
            type='AnchorGenerator',
            octave_base_scale=4,                        # faster: None.
            scales_per_octave=3,                        # faster: None.
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),              # faeter: 4 ~ 64.
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(                                  # [CRITICAL] CELoss -> FocalLoss.
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_cnt=dict(
            type='CrossEntropyLoss',
            loss_weight=0.1)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssignerWithCount',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,                            # faster: 0.5
            min_pos_iou=0,                              # faster: 0.5
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
