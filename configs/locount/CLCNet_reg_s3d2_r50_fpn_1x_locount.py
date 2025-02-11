_base_ = [
    'CLCNet_cls_s3d2_r50_fpn_1x_locount.py'
]
model = dict(
    roi_head=dict(
        count_loss_weights=[0.1, 0.1, 0.1],
        bbox_head=[
            dict(
                type='FCBBoxHeadWithCount',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=140,
                num_counts=56,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                reg_count_strategy=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0),
                loss_cnt=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='FCBBoxHeadWithCount',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=140,
                num_counts=56,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_count_strategy=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0),
                loss_cnt=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='FCBBoxHeadWithCount',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=140,
                num_counts=56,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                reg_count_strategy=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0),
                loss_cnt=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0))
        ]
    )
)