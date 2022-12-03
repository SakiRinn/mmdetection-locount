# 工作表
## 数据集
* `tools/misc/locount_txt2json.py`\
  将原为`.txt`格式的annotations转化为单个`.json`文件。
* `mmdet/datasets/pipelines/loading.py`\
  主要修改`LoadAnnotations`类。
  添加`with_count`参数，并实现了`_load_count`函数。
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
* `mmdet/models/detectors/cascade_rcnn.py`\
  添加`CascadeRCNNWithCount`类。
  具体改动为在`forward_train`函数下添加`gt_counts`参数。
  ```Python
  # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
  #                                          gt_bboxes, gt_labels,
  #                                          gt_bboxes_ignore, gt_masks,
  #                                          **kwargs)
  roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                           gt_bboxes, gt_labels, gt_counts,
                                           gt_bboxes_ignore, gt_masks,
                                           **kwargs)
  ```
* `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`\
  添加`FCBBoxHeadWithCount`类。
* `mmdet/models/roi_heads/cascade_roi_head.py`\
  添加`CascadeRoIHeadWithCount`类。
* `mmdet/models/roi_heads/__init__.py`\
  补充新添加的类至`__all__`.

## 数据采样
* `mmdet/core/bbox/samplers/random_sampler.py`\
  添加`RandomSamplerWithCount`类。
* `mmdet/core/bbox/samplers/sampling_result.py`\
  添加`SamplingResultWithCount`类。
* `mmdet/core/bbox/assigners/max_iou_assigner.py`\
  添加`MaxIoUAssignerWithCount`类。
* `mmdet/core/bbox/assigners/assign_result.py`\
  添加`AssignResultWithCount`类。

## 配置文件
* `configs/_base_/datasets/cocoLHC.py`\
  新文件，用于处理Locount数据集。
* `configs/_base_/models/cascade_rcnn_r50_fpn_Locount.py`\
  新文件，用于初始化Locount网络结构。
* `configs/locount/cascade_rcnn_r50_fpn_1x_Locount.py`\
  新文件，用于整合。