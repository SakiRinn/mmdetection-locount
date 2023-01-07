# 工作表
## 数据集
该部分修改主要在`mmdet/datasets/`下进行。
* `tools/misc/locount_txt2json.py`\
  将原为`.txt`格式的annotations转化为单个`.json`文件。
* `mmdet/datasets/pipelines/loading.py`\
  主要修改`LoadAnnotations`类。\
  添加`with_count`参数，并实现了`_load_count`函数。
* `mmdet/datasets/pipelines/formatting.py`\
  主要修改`ToDataContainer`类。\
  修改`__init__`函数中`field`参数的默认值，添加了`gt_counts`键。
* `mmdet/datasets/coco.py`\
  检测条件修改，删除了`area`键的检测。
  ```Python
  # if ann['area'] <= 0 or w < 1 or h < 1:
  if w < 1 or h < 1:
  ```
* `mmdet/datasets/pipelines/formatting.py`\
  将`gt_counts`加入至使用`DataContainer`嵌套的标签。
  ```Python
  # for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
  for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_counts']:
      if key not in results:
          continue
      results[key] = DC(to_tensor(results[key]))
  ```

## 模型
该部分修改主要在`mmdet/models/`下进行。
* `mmdet/models/detectors/two_stage.py`\
  **基类**。添加`TwoStageDetectorWithCount`类。\
  具体为，在`forward_train`函数里，调用`roi_head`的`forward_train`时添加`gt_count`变量。
* `mmdet/models/detectors/cascade_rcnn.py`\
  添加`CascadeRCNNWithCount`类，具体同上。
* `mmdet/models/roi_heads/bbox_heads/bbox_head.py`\
  **基类**。添加`BBoxHeadWithCount`类。大幅修改原类。
* `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`\
  添加`FCBBoxHeadWithCount`类，具体添加了一个cnt头。
* `mmdet/models/roi_heads/cascade_roi_head.py`\
  添加`CascadeRoIHeadWithCount`类，具体添加了cnt有关部分。

## 数据采样
该部分修改主要在`mmdet/core/`下进行。
* `mmdet/core/bbox/assigners/base_sampler.py`\
  **基类**。添加`BaseSamplerWithCount`类，添加了`gt_counts`变量。
* `mmdet/core/bbox/assigners/base_assigner.py`\
  **基类**。添加`BaseAssignerWithCount`类，添加了`gt_counts`变量。
* `mmdet/core/bbox/samplers/random_sampler.py`\
  添加`RandomSamplerWithCount`类。
* `mmdet/core/bbox/samplers/sampling_result.py`\
  添加`SamplingResultWithCount`类。
* `mmdet/core/bbox/assigners/max_iou_assigner.py`\
  添加`MaxIoUAssignerWithCount`类。
* `mmdet/core/bbox/assigners/assign_result.py`\
  添加`AssignResultWithCount`类。
* `mmdet/core/bbox/transforms.py`\
  `bbox2result`函数中添加`counts`参数。
* `mmdet/core/evaluation/class_names.py`\
  修改`coco_classes`中的类型。
* `mmdet/core/post_processing/bbox_nms.py`\
  添加`multiclass_nms_with_count`函数。

## 配置文件
该部分修改主要在`configs/`下进行。
* `configs/_base_/datasets/coco_LHC.py`\
  **新文件**。用于处理Locount数据集。
* `configs/_base_/models/cascade_rcnn_r50_fpn_locount.py`\
  **新文件**。用于初始化Locount网络结构。
* `configs/locount/cascade_rcnn_r50_fpn_1x_locount.py`\
  **新文件**。用于整合。

# 启动
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

## 现有问题
疑似cuda越界，报错如下
```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [0,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [1,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [2,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [3,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [4,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [5,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [6,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [7,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [8,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [9,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [10,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [11,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [12,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [13,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [14,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [15,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [16,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [17,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [18,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [19,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [20,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [21,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [22,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [23,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [24,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [25,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [26,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [27,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [28,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [29,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [30,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [31,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [32,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [33,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [34,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [35,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [36,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [37,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [38,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [39,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [40,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [41,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [42,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [43,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [44,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [45,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [46,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [47,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [48,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [49,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [50,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [51,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [52,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [53,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [54,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [55,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [56,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [57,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [58,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [59,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [60,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [61,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [62,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [63,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [64,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [65,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [66,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [67,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [68,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [69,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [70,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [71,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [72,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [73,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [74,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [75,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [76,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [77,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [78,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [79,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [80,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [81,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [82,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [83,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [84,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [85,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [86,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [87,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [88,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [89,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [90,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [91,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [92,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [93,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [94,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [95,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [96,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [97,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [98,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [99,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [100,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [101,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [102,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [103,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [104,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [105,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [106,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [107,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [108,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [109,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [110,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [111,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [112,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [113,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [114,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [115,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [116,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [117,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [118,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [119,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [120,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [121,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [122,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [123,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [124,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [125,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [126,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:84: operator(): block: [0,0,0], thread: [127,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
Traceback (most recent call last):
  File "tools/train.py", line 244, in <module>
    main()
  File "tools/train.py", line 233, in main
    train_detector(
  File "/root/cms/mmdetection/mmdet/apis/train.py", line 246, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 136, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 53, in train
    self.run_iter(data_batch, train_mode=True, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 31, in run_iter
    outputs = self.model.train_step(data_batch, self.optimizer,
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/parallel/data_parallel.py", line 77, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/root/cms/mmdetection/mmdet/models/detectors/base.py", line 248, in train_step
    losses = self(**data)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 119, in new_func
    return old_func(*args, **kwargs)
  File "/root/cms/mmdetection/mmdet/models/detectors/base.py", line 172, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/root/cms/mmdetection/mmdet/models/detectors/cascade_rcnn.py", line 104, in forward_train
    roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
  File "/root/cms/mmdetection/mmdet/models/roi_heads/cascade_roi_head.py", line 760, in forward_train
    assign_result = bbox_assigner.assign(
  File "/root/cms/mmdetection/mmdet/core/bbox/assigners/max_iou_assigner.py", line 258, in assign
    overlaps = self.iou_calculator(gt_bboxes, bboxes)
  File "/root/cms/mmdetection/mmdet/core/bbox/iou_calculators/iou2d_calculator.py", line 65, in __call__
    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
  File "/root/cms/mmdetection/mmdet/core/bbox/iou_calculators/iou2d_calculator.py", line 250, in bbox_overlaps
    eps = union.new_tensor([eps])
RuntimeError: CUDA error: device-side assert triggered
terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: device-side assert triggered
Exception raised from create_event_internal at /pytorch/c10/cuda/CUDACachingAllocator.cpp:687 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x42 (0x7f6801c4f8b2 in /root/miniconda3/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::CUDACachingAllocator::raw_delete(void*) + 0xad2 (0x7f6801ea1952 in /root/miniconda3/lib/python3.8/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10::TensorImpl::release_resources() + 0x4d (0x7f6801c3ab7d in /root/miniconda3/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #3: <unknown function> + 0x5fa0a2 (0x7f6804b580a2 in /root/miniconda3/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #4: <unknown function> + 0x5fa156 (0x7f6804b58156 in /root/miniconda3/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
<omitting python frames>
frame #23: __libc_start_main + 0xe7 (0x7f680e260b97 in /lib/x86_64-linux-gnu/libc.so.6)

Aborted (core dumped)
```