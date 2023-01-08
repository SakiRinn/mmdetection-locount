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
* `mmdet/models/dense_heads/anchor_head.py`\
  **基类**。添加`AnchorHeadWithCount`类。大幅修改原类。
* `mmdet/models/dense_heads/rpn_head.py`\
  添加`RPNHeadWithCount`类，具体添加了cnt有关部分。
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
shape没对上？
```
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
  File "/root/cms/mmdetection/mmdet/models/detectors/two_stage.py", line 253, in forward_train
    rpn_losses, proposal_list = self.rpn_head.forward_train(
  File "/root/cms/mmdetection/mmdet/models/dense_heads/base_dense_head.py", line 335, in forward_train
    losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
  File "/root/cms/mmdetection/mmdet/models/dense_heads/rpn_head.py", line 323, in loss
    losses = super(RPNHeadWithCount, self).loss(
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 208, in new_func
    return old_func(*args, **kwargs)
  File "/root/cms/mmdetection/mmdet/models/dense_heads/anchor_head.py", line 878, in loss
    cls_reg_cnt_targets = self.get_targets(
  File "/root/cms/mmdetection/mmdet/models/dense_heads/anchor_head.py", line 788, in get_targets
    results = multi_apply(
  File "/root/cms/mmdetection/mmdet/core/utils/misc.py", line 30, in multi_apply
    return tuple(map(list, zip(*map_results)))
  File "/root/cms/mmdetection/mmdet/models/dense_heads/anchor_head.py", line 750, in _get_targets_single
    counts = unmap(
  File "/root/cms/mmdetection/mmdet/core/utils/misc.py", line 38, in unmap
    ret[inds.type(torch.bool)] = data
RuntimeError: shape mismatch: value tensor of shape [257796] cannot be broadcast to indexing result of shape [226128]
```