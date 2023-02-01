_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_locount.py',
    '../_base_/datasets/coco_LHC.py',
    '../_base_/schedules/schedule_1x_locount.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))