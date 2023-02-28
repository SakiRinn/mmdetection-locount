_base_ = [
    'faster_rcnn_r50_fpn_1x_locount.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_count_strategy=True,
            loss_cnt=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=0.1)
        )
    )
)