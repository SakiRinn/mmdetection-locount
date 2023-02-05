# 工作表

## 数据集
该部分修改主要在`mmdet/datasets/`下进行。
全部已完工，可复用。

### `datasets/pipelines`部分
* `mmdet/datasets/pipelines/loading.py`\
  主要修改`LoadAnnotations`类。\
  添加`with_count`参数，并实现了`_load_count`函数。
* `mmdet/datasets/pipelines/formatting.py`\
  主要修改`ToDataContainer`类。\
  修改`__init__`函数中`field`参数的默认值，添加了`gt_counts`键。
* `mmdet/datasets/pipelines/formatting.py`\
  将`gt_counts`加入至使用`DataContainer`嵌套的标签。
  ```Python
  # for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
  for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_counts']:
      if key not in results:
          continue
      results[key] = DC(to_tensor(results[key]))
  ```

### 其它
* `cntcocotools`包
  重写了`pytotocools`包，添加了count相关部分。
* `cntcocotools/txt2json.py`\
  **新文件**。将原为`.txt`格式的annotations转化为单个`.json`文件。
* `mmdet/datasets/coco.py`\
  修改`CLASSES`常量与`coco_classes`一致，以及`_parse_ann_info`函数。


## 模型
该部分修改主要在`mmdet/models/`下进行。

### `models/detectors`部分
已完工，仅基类可复用。
* `mmdet/models/detectors/two_stage.py`\
  **基类**。添加`TwoStageDetectorWithCount`类。\
  具体为，在`forward_train`函数里，调用`roi_head`的`forward_train`时添加`gt_count`变量。
* `mmdet/models/detectors/single_stage.py`\
  **基类**。添加`SingleStageDetectorWithCount`类。
* `mmdet/models/detectors/cascade_rcnn.py`\
  添加`CascadeRCNNWithCount`类，仅修改了基类。
* `mmdet/models/detectors/faster_rcnn.py`
  添加`FasterRCNNWithCount`类，仅修改了基类。
* `mmdet/models/detectors/faster_rcnn.py`
  添加`RetinaNetWithCount`类，仅修改了基类。

### `models/dense_heads`部分
已完工，仅基类可复用。
适用于单stage结构。对于多stage的RPN部分，仅使用不带count的dense head即可。
* `mmdet/models/dense_heads/base_dense_head.py`\
  **基类**。添加`BaseDenseHeadWithCount`类。
* `mmdet/models/dense_heads/anchor_head.py`\
  添加`AnchorHeadWithCount`类。
* `mmdet/models/dense_heads/retina_head.py`
  添加`RetinaHeadWithCount`类。

### `models/roi_heads`部分
已完工，仅基类可复用。
* `mmdet/models/roi_heads/bbox_heads/bbox_head.py`\
  **基类**。添加`BBoxHeadWithCount`类。\
  实现了随stage数count预测从模糊到精确的过程，具体为`div_counts`函数和`coarse_counts`属性（替代原`num_counts`）。
* `mmdet/models/roi_heads/test_mixins.py`\
  **基类**。添加`BBoxTestMixinWithCount`类，为RoI Head的派生类提供test相关方法。
* `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`\
  添加`FCBBoxHeadWithCount`类，具体添加了一个cnt头。
* `mmdet/models/roi_heads/cascade_roi_head.py`\
  添加`CascadeRoIHeadWithCount`类。
* `mmdet/models/roi_heads/standard_roi_head.py`\
  添加`StandardRoIHeadWithCount`类。


## 数据采样
该部分修改主要在`mmdet/core/bbox/`下进行。
全部已完工，可复用。

### `core/bbox/assigners`部分
用于指定正负样本。
* `mmdet/core/bbox/assigners/base_assigner.py`\
  **基类**。添加`BaseAssignerWithCount`类，添加了`gt_counts`变量。
* `mmdet/core/bbox/assigners/max_iou_assigner.py`\
  添加`MaxIoUAssignerWithCount`类。
* `mmdet/core/bbox/assigners/assign_result.py`\
  添加`AssignResultWithCount`类。

### `core/bbox/samplers`部分
用于指定后对样本采样。
* `mmdet/core/bbox/assigners/base_sampler.py`\
  **基类**。添加`BaseSamplerWithCount`类，添加了`gt_counts`变量。
* `mmdet/core/bbox/samplers/random_sampler.py`\
  添加`RandomSamplerWithCount`类。
* `mmdet/core/bbox/samplers/sampling_result.py`\
  添加`SamplingResultWithCount`类。

### 其它
* `mmdet/core/bbox/transforms.py`\
  `bbox2result`函数中添加`counts`参数。


## 配置文件
该部分修改主要在`configs/`下进行。
* `configs/_base_/datasets/coco_LHC.py`\
  **新文件**。用于处理Locount数据集。
* `configs/_base_/models/*_locount.py`\
  **新文件**。用于初始化Locount网络结构。（多个文件）
* `configs/locount/`\
  **新文件夹**。用于整合locount配置。

# 备注
## 生成数据集配置
单独运行`tools/misc/locount_txt2json.py`中的`txt2json`函数，将得到的json文件放到数据文件夹。

分别对训练集和测试集使用，得到`test.json`和`train.json`文件。

```Python
if __name__ == '__main__':
    txt2json('data/Locount/annotation_train', 'test.train')
    txt2json('data/Locount/annotation_test', 'test.test')
```

## 训练
运行`tools/train.py`脚本，指定一个config文件。
```sh
python tools/train.py configs/locount/cascade_rcnn_r50_fpn_1x_locount.py
```

## 推理
运行`tools/test.py`脚本，指定一个config文件和一个checkpoint.
```sh
python tools/test.py configs/locount/cascade_rcnn_r50_fpn_1x_locount.py work_dirs/cascade_rcnn_r50_fpn_1x_locount/latest.pth
```
