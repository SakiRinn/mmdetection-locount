_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_locount.py',
    '../_base_/datasets/coco_LHC.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        num_stages=2,
        stage_loss_weights=[1, 0.5],
        stage_cnt_loss_weights=[0.1, 0.1],
        base=10,
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
                    use_sigmoid=False,
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
                    use_sigmoid=False,
                    loss_weight=1.0)),
        ]
    )
)