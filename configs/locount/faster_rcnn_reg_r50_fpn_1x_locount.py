_base_ = [
    'faster_rcnn_r50_fpn_1x_locount.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='FCBBoxHeadWithCount',
            reg_count_strategy=True,
            loss_cnt=dict(
                type='L1Loss',
                loss_weight=0.1
            )
        )
    )
)