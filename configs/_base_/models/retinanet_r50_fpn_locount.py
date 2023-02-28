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
        start_level=1,                                  # faster: start_level=0, add_extra_convs='on_output'.
        add_extra_convs='on_input',                     # where the input for the extra convs comes from. (on input, literal, output)
        num_outs=5),
    bbox_head=dict(
        type='RetinaHeadWithCount',
        num_classes=140,
        num_counts=56,
        in_channels=256,                                #                                    /-> reg_conv(feat_chns, base_anchors*4)
        stacked_convs=4,                                # N*stacked_conv(in_chns, feat_chns) --> cls_conv(feat_chns, base_anchors*num_classes)
        feat_channels=256,
        anchor_generator=dict(                          # faeter: in `rpn_head`, scales=[8].
            type='AnchorGenerator',
            octave_base_scale=4,                        # scales = octave_base_scale * octave_scales   e.g. 4*[2**0, 2**(1/3), 2**(2/3)]
            scales_per_octave=3,                        # octave_scales = [2**(i / scales_per_octave) for i in range(scales_per_octave)]
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),              # faeter: 4 ~ 64.
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(                                  # [CRITICAL] CELoss -> FocalLoss
            type='FocalLoss',                           # [CRITICAL] Binary classification loss! So use sigmoid.
            use_sigmoid=True,                           # Whether `x = sigmoid(x)` before calc.
            gamma=2.0,                                  # alpha * (1-p)**gamma * log(p)
            alpha=0.25,                                 # gamma: adjust weight of **easy** samples, alpha: weight of **pos** samples.
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
            pos_iou_thr=0.5,                            # faster: 0.7
            neg_iou_thr=0.4,                            # faster: 0.3
            min_pos_iou=0,                              # faster: 0.3
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
