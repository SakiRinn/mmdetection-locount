_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_locount.py',
    '../_base_/datasets/coco_LHC.py',
    '../_base_/schedules/schedule_1x_locount.py',
    '../_base_/default_runtime.py'
]
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)