# Locount: Exploring a Novel Object Detection Solution in Retail Scenarios

This project is the source code of the paper "Locount: Exploring a Novel Object Detection Solution in Retail Scenarios", based on [MMDetection](https://github.com/open-mmlab/mmdetection) v2.26 (commit id: `31c84958f54287a8be2b99cbf87a6dcf12e57753`).


## Quick Start

This project didn't destroy the original usage of MMDetection, which means that you can use this project with full reference to [the documentation of MMDetection v2.26](https://mmdetection.readthedocs.io/en/v2.26.0/).

This section only explains the most basic usage of the project. For more detailed instructions, please refer to [the documentation of MMDetection v2.26](https://mmdetection.readthedocs.io/en/v2.26.0/).

### Generate the annotation of Locount

We have coded the `txt2json` function in the file `tools/misc/locount_txt2json.py` to facilitate you to generate COCO format annotation from the Locount dataset.

Use it on the train set and test set respectively to get the `test.json` and `train.json` files.

```Python
if __name__ == '__main__':
    txt2json('data/Locount/annotation_train', 'train.json')
    txt2json('data/Locount/annotation_test', 'test.json')
```

After that, in the `data/` folder of the root directory, store the Locount data set according to the following file structure:

```
data/
└── Locount
    ├── annotation_test
    ├── annotation_train
    ├── Locount_ImagesTest
    ├── Locount_ImagesTrain
    ├── test.json
    └── train.json
```

### Train

Run the `tools/train.py` script with a config file.

All model configuration files related to Locount are stored in the `configs/locount/` directory.

Example:
```sh
python tools/train.py configs/locount/cascade_rcnn_r50_fpn_1x_locount.py
```

### Test

Run the `tools/test.py` script with a config file and a checkpoint file.

The config file used in testing should be the same as the one used in training.

Example:
```sh
python tools/test.py configs/locount/cascade_rcnn_r50_fpn_1x_locount.py work_dirs/cascade_rcnn_r50_fpn_1x_locount/latest.pth
```


## Download

### Locount Dataset

[Baidu Link](https://pan.baidu.com/s/1WTzOr3metW-BTBFdslgFUg?pwd=e24y) Updated 2021/04/06

Password: e24y

x1,y1,x2,y2,cls,cnt

(x1,y1) : Coordinate of the upper left corner of the rectangle surrounding box.

(x2,y2) : coordinate of the lower right corner of the rectangle surrounding box.

CLS: The category of the target in the rectangular bounding box.

CNT: The number of instances in the rectangle.

### Trained models and results

[Baidu Link](https://pan.baidu.com/s/1W-w-N61JVchkTgrjqnjo6A?pwd=j57u) Updated 2023/04/13

Password: j57u

The files ending with `.pth` are the checkpoints of the model.

The files ending with `.bbox.json` are the bbox sets inferred by the model on the test set.


## Results

We set `maxDet=150` to measure the performance with and without count.

For AP/AR, the IoU threshold and AC threshold (if with count) range from 0.5 to 0.95.

For AP_`num`, `num` refer to the threshold of IoU and AC (if with count).

### With count

| Model                    | AP   | AP_0.5 | AP_0.75 | AR   |
| ------------------------ | ---- | ------ | ------- | ---- |
| Faster RCNN  (cls)       | 41.5 | 60.8   | 49.1    | 52.0 |
| Cascade RCNN (cls-s2d10) | 43.5 | 60.5   | 50.7    | 53.0 |
| Cascade RCNN (cls-s3d2)  | 43.5 | 60.6   | 51.1    | 53.1 |
| Cascade RCNN (cls-s6d2)  | 43.0 | 59.2   | 50.2    | 51.7 |
| Faster RCNN  (reg)       | 41.6 | 60.6   | 49.1    | 51.8 |
| Cascade RCNN (reg-s2)    | 43.3 | 60.6   | 50.7    | 53.0 |
| Cascade RCNN (reg-s3)    | 43.0 | 60.2   | 50.5    | 52.9 |
| Cascade RCNN (reg-s6)    | 42.7 | 59.4   | 50.1    | 52.1 |
| FCOS                     | 39.4 | 57.9   | 45.9    | 57.8 |
| RetinaNet                | 39.4 | 56.7   | 46.2    | 56.1 |
| Reppoints                | 39.7 | 56.8   | 46.3    | 56.7 |
| DDOD                     | 44.8 | 63.9   | 51.6    | 60.6 |
| PAA                      | 40.7 | 58.1   | 47.1    | 58.4 |
| PAA (3x)                 | 45.0 | 64.2   | 52.1    | 60.5 |
| Sparse RCNN              | 44.9 | 63.7   | 52.0    | 62.2 |
| Sparse RCNN (3x)         | 48.3 | 67.8   | 56.2    | 70.2 |
| Deformable DETR          | 48.3 | 68.2   | 56.6    | 63.3 |
| Deformable DETR (s2)     | 49.3 | 69.5   | 57.3    | 63.2 |

### Without count

| Model                    | AP   | AP_0.5 | AP_0.75 | AR   |
| ------------------------ | ---- | ------ | ------- | ---- |
| Faster RCNN  (cls)       | 44.6 | 63.4   | 52.5    | 54.5 |
| Cascade RCNN (cls-s2d10) | 47.1 | 63.6   | 54.9    | 56.3 |
| Cascade RCNN (cls-s3d2)  | 47.1 | 63.6   | 55.1    | 56.2 |
| Cascade RCNN (cls-s6d2)  | 46.6 | 62.2   | 54.4    | 54.9 |
| Faster RCNN  (reg)       | 45.3 | 64.0   | 53.3    | 55.8 |
| Cascade RCNN (reg-s2)    | 47.1 | 63.7   | 55.0    | 56.5 |
| Cascade RCNN (reg-s3)    | 47.0 | 63.5   | 55.0    | 56.4 |
| Cascade RCNN (reg-s6)    | 46.8 | 62.8   | 54.8    | 55.7 |
| FCOS                     | 42.4 | 60.5   | 49.2    | 61.0 |
| RetinaNet                | 42.6 | 59.4   | 49.7    | 59.5 |
| Reppoints                | 43.2 | 59.9   | 50.1    | 60.5 |
| DDOD                     | 48.4 | 67.1   | 55.7    | 64.2 |
| PAA                      | 45.8 | 63.8   | 52.9    | 63.4 |
| PAA (3x)                 | 50.1 | 70.0   | 57.9    | 65.4 |
| Sparse RCNN              | 48.2 | 66.4   | 55.8    | 65.5 |
| Sparse RCNN (3x)         | 51.9 | 70.8   | 60.4    | 73.7 |
| Deformable DETR          | 52.1 | 71.2   | 60.9    | 67.1 |
| Deformable DETR (s2)     | 53.3 | 73.2   | 61.8    | 67.0 |


## Changes

This project is based on MMDetection v2.26, modified and added new classes (names usually ending with `WithCount`).

The overall structure of this project is consistent with the original MMDetection. This section lists the specific changes compared to the original MMDetection.

### Dataset

Mainly modify the `CocoDataset` class at the `mmdet/datasets/` directory.

Our dataset "Locount" follows the COCO format, so modify the code directly on the `CocoDataset` class.

Add the `RandomShiftWithCount` and `RandomCropWithCount` classes to the file `mmdet/datasets/pipelines/transforms.py`.

In the file `mmdet/datasets/coco.py`, make the `CLASSES` constant consistent with `coco_classes` and modify the `_parse_ann_info` function.

### Pipeline

Mainly modify code files at the `mmdet/datasets/pipelines/` directory.

The modifications are as follows:

* `mmdet/datasets/pipelines/loading.py`\
  Mainly modify the `LoadAnnotations` class.\
  Added the `with_count` parameter and implemented the `_load_count` function.
* `mmdet/datasets/pipelines/formatting.py`\
  Modify the `ToDataContainer` class, modify the default value of the `field` parameter in the `__init__` function, and add the `gt_counts` key.

### Models

Mainly modify code files at the `mmdet/models/` directory.

The modifications are as follows:

* `models/detectors/`\
  Add base classes `TwoStageDetectorWithCount`, `SingleStageDetectorWithCount`, and some model classes whose names end with `WithCount`.
* `models/dense_heads/`\
  Add base classes `BaseDenseHeadWithCount`, `AnchorFreeHeadWithCount` and `AnchorHeadWithCount`, and some header classes whose names end with `WithCount`.
  Only suitable for single stage structure. For the multi-stage RPN part, use the dense head without count.
* `models/roi_heads/`\
  Add base classes `BBoxHeadWithCount`, `BBoxTestMixinWithCount`, and some header classes whose names end with `WithCount`.
  In the `BBoxHeadWithCount` class, the process of predicting the number of stages from fuzzy to precise is implemented, specifically the `div_counts` function and the `coarse_counts` attribute (replacing the original `num_counts`).


### Data Sampling

Mainly modify code files at the `mmdet/core/bbox/` directory.

The modifications are as follows:

* Assigners
  Used to determine positive and negative samples, mainly modifying files at the `mmdet/core/bbox/assigners/` directory.
  Add `MaxIoUAssignerWithCount`, `AssignResultWithCount`, `PointAssignerWithCount`, `UniformAssignerWithCount`, `ATSSAssignerWithCount`, `HungarianAssignerWithCount`, `BaseAssignerWithCount` classes, where `BaseAssignerWithCount` class is the base class of all assigners.
  Added `gt_counts` variable in file `base_assigner.py`.
* Samplers
  Used to sample after specification, mainly modifying files at the `mmdet/core/bbox/samplers/` directory.
  Add `RandomSamplerWithCount`, `SamplingResultWithCount`, `BaseSamplerWithCount`, `PesudoSamplerWithCount`, `RandomSamplerWithCount` classes, where `BaseSamplerWithCount` class is the base class of all samplers.
  Added `gt_counts` variable in file `base_sampler.py`.

* Others
  Add the `counts` parameter to the `bbox2result` function in the file `mmdet/core/bbox/transforms.py`.

### Configs

Mainly modify code files at the `configs/` directory, the following new files have been added:

* `configs/_base_/datasets/coco_LHC.py`\
  Used to process the Locount data set.
* `configs/_base_/models/*_locount.py`\
  Used to initialize the Locount model structure.
* `configs/locount/`\
  Used to integrate locount configuration.

### Others

* `cntcocotools/`\
  Rewrite the `pytotocools` package to add the count-related parts.

* `cntcocotools/txt2json.py`\
  Convert annotations originally in `.txt` format into a single `.json` file.
