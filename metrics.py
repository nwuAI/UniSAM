import torch
import numpy as np
import cv2
from scipy import ndimage
import numpy as np

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y

def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def wFmeasure(FG, GT):
    """
    wFmeasure Compute the Weighted F-beta measure (as proposed in "How to Evaluate
    Foreground Maps?" [Margolin et. al - CVPR'14])

    Arguments:
        FG (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT (np.ndarray): Binary ground truth. Type: bool
    Return:
        float : The Weighted F-beta score
    """
    if not GT.max():
        return 0
    FG = FG.detach().cpu().numpy() # 假设FG是一个PyTorch张量
    GT = GT.detach().cpu().numpy()  # 假设GT也是一个PyTorch张量

    E = np.abs(FG - GT)

    Dst, IDXT = ndimage.distance_transform_edt(1 - GT.astype(np.float64), return_indices=True)
    # Pixel dependency
    Et = E.copy()
    Et[np.logical_not(GT)] = Et[IDXT[0][np.logical_not(GT)], IDXT[1][np.logical_not(GT)]]  # To deal correctly with the edges of the foreground region
    EA = ndimage.gaussian_filter(Et, 5, mode='constant', truncate=0.5)
    MIN_E_EA = E.copy()
    MIN_E_EA[np.logical_and(GT, EA < E)] = EA[np.logical_and(GT, EA < E)]
    # Pixel importance
    B = np.ones(GT.shape)
    B[np.logical_not(GT)] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5 * Dst[np.logical_not(GT)])
    Ew = MIN_E_EA * B

    TPw = GT.sum() - Ew[GT].sum()
    FPw = Ew[np.logical_not(GT)].sum()

    R = 1 - Ew[GT].mean()  # Weighted Recall
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)  # Weighted Precision

    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)  # Beta=1
    # Q = (1 + Beta ** 2) * R * P / (R + Beta * P + np.finfo(np.float64).eps)

    return Q

def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'mae':
            metric_list.append(np.mean(mae(pred, label)))
        elif metric == 'ber':
            metric_list.append(np.mean(ber(pred, label)))
        elif metric == 'Fβ':
            metric_list.append(np.mean(wFmeasure(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric

def iou(pr, gt, eps=1e-20, threshold = 0.5):
    pr = torch.sigmoid(pr)#.cpu()
    pr_ = (pr >= 0.5)
    gt_ = (gt >= 0.5)
    iou = (torch.sum((pr_ & gt_)) / torch.sum((pr_ | gt_))).cpu().numpy()

    return iou

def mae(pr, gt, eps=1e-20, threshold = 0.5):
    pr = torch.sigmoid(pr)#.cpu()
    pr_ = torch.where(pr >= 0.5, torch.ones_like(pr), torch.zeros_like(pr))
    gt_ = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))
    mae = torch.abs(pr_ - gt_).mean().cpu().numpy()

    return mae

def ber(pr, gt, eps=1e-20, threshold = 0.5):
    pr = torch.sigmoid(pr)#.cpu()
    pr_ = (pr >= 0.5)
    gt_ = (gt >= 0.5)
    N_p = torch.sum(gt_) + 1e-20
    N_n = torch.sum(torch.logical_not(gt_)) + 1e-20  # should we add this？
    TP = torch.sum(pr_ & gt_)
    TN = torch.sum(torch.logical_not(pr_) & torch.logical_not(gt_))
    ber = (1 - (1 / 2) * ((TP / N_p) + (TN / N_n)))

    if not torch.isnan(ber):
        return ber.item() * 100
    else:
        return 0

def SegMetrics_Glass(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'mae':
            metric_list.append(np.mean(mae(pred, label)))
        elif metric == 'ber':
            metric_list.append(np.mean(ber(pred, label)))
        elif metric == 'Fβ':
            metric_list.append(np.mean(wFmeasure(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric