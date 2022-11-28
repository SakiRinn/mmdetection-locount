# 工作表
## 数据集
* `tools/misc/locount_txt2json.py`\
  将原为`.txt`格式的annotations转化为单个`.json`文件。
* `mmdet/datasets/pipelines/loading.py`\
  添加`with_count`参数，并实现了`_load_count`函数。
* `mmdet/datasets/coco.py`\
  检测条件修改，删除了`area`键的检测。
  ```Python
  # if ann['area'] <= 0 or w < 1 or h < 1:
  if w < 1 or h < 1:
  ```
## 模型
* `mmdet/models/detectors/two_stage.py`\
  在`forward_train`函数下添加`gt_counts`参数，并添加计数部分代码。
## 配置文件
* `configs/_base_/datasets/cocoLHC.py`\
  新文件，用于处理Locount数据集。
  预处理部分添加`gt_counts`键。
* `configs/_base_/models/cascade_rcnn_r50_fpn_Locount.py`\
  新文件，用于初始化Locount网络结构。
* `configs/locount/cascade_rcnn_r50_fpn_1x_Locount.py`\
  新文件，用于整合。