# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn


@mmcv.jit(coderize=True)
def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

@mmcv.jit(coderize=True)
def cnt_accuracy(pred, count, reduce_mean=True):
    """Calculate counting accuracy (AC).

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class) or (N, ).
        count (torch.Tensor): The count target of each prediction, shape (N, ).

    Returns:
        float | np.ndarray
    """
    if pred.size(0) == 0:
        ac = pred.new_tensor(0.)
        return ac
    assert pred.ndim <=2 and count.ndim == 1
    assert pred.size(0) == count.size(0)

    if pred.ndim == 2:
        _, pred_count = pred.max(dim=1)
    else:
        pred_count = pred
    ac = (1 - torch.abs(pred_count - count) / count).clamp(min=0)
    if reduce_mean:
        return ac.mean()
    return ac

def single_cnt_accuracy(pred_count, count):
    """Calculate counting accuracy (AC) for a single number.
       Using native python.

    Args:
        pred (int): The model prediction.
        count (int): The count target of each prediction.

    Returns:
        float
    """
    assert isinstance(pred_count, int) and isinstance(count, int)
    return max(.0, 1.0 - abs(pred_count - count) / count)


class Accuracy(nn.Module):

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
