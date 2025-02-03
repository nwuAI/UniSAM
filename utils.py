from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as F1
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
from skimage import measure
import torch.nn as nn
import logging
import os
import time
import sys
import collections
from PIL import Image


def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if mask.dim()==4:
        mask = mask.squeeze()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    label_img = label(mask)
    regions = regionprops(label_img)

    boxes = [tuple(region.bbox) for region in regions]

    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)

        if len(boxes) == 0:
            default_box = (0, 0, 1, 1)
            boxes = [default_box] * num_duplicates

        elif len(boxes) > 0:
            boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)

def get_boxes_from_mask_glass(mask, box_num=1, std=0.1, max_pixel=3):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        boxes = [(0, 0, 1, 1)]
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
        # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)

def select_random_points(pr, gt, point_num = 9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    if pr.dim()==2:
        pr = pr.unsqueeze(0)
    if gt.dim()==2:
        gt = gt.unsqueeze(0)
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []

    for j in range(error.shape[0]):
        one_pred = pred[j]#.squeeze(0)
        one_gt = gt[j]#.squeeze(0)
        one_erroer = error[j]#.squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []

        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 0:
                label = 0
            elif one_pred[x,y] == 1 and one_gt[x,y] == 1:
                label = 1
            else:
                label = -1
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling_glass(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg

        if fg_size > 0:
            fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
            fg_coords_sampled = fg_coords[fg_indices]
        else:
            fg_coords_sampled = np.empty((0, 2))

        if bg_size > 0:
            bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
            bg_coords_sampled = bg_coords[bg_indices]
        else:
            bg_coords_sampled = np.empty((0, 2))

        coords = np.concatenate([fg_coords_sampled, bg_coords_sampled], axis=0)
        labels = np.concatenate([np.ones(len(fg_coords_sampled)), np.zeros(len(bg_coords_sampled))]).astype(int)
        indices = np.random.permutation(len(coords))
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels

def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    if isinstance(mask, torch.Tensor):
        # mask = mask.numpy()
        mask = mask.cpu().numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size[0] and ori_w < img_size[1]:
        transforms.append(A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size[0]), int(img_size[1]), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    # return A.Compose(transforms, p=1., additional_targets={'depth': 'image', 'mask': 'mask'})
    return A.Compose(transforms, p=1., additional_targets={'depth': 'image'})

def train_transforms_aug(img_size, ori_h, ori_w):
    cfg = {
        "scales_range": "0.5 2.0",
        "crop_size": "416 416",
        'brightness': 0.5,
        'contrast': 0.5,
        'saturation': 0.5,
        'p': 0.5  # 水平翻转的概率
    }

    scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
    crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))
    transforms = []

    # 如果图片尺寸小于目标尺寸，先进行填充操作
    if ori_h < img_size[0] and ori_w < img_size[1]:
        transforms.append(
            A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT,
                          value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size[0]), int(img_size[1]), interpolation=cv2.INTER_NEAREST))

    # 色彩增强
    transforms.append(A.ColorJitter(brightness=cfg['brightness'], contrast=cfg['contrast'], saturation=cfg['saturation'], hue=0.2, p=0.5))

    # 随机水平翻转
    transforms.append(A.HorizontalFlip(p=0.5))

    # 随机旋转
    transforms.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=0.7))

    # 随机缩放
    # transforms.append(A.RandomScale(scale_limit=scale_range))

    # 随机裁剪
    transforms.append(A.RandomCrop(height=img_size[0], width=img_size[1], p=1))

    # 转换为张量
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, p=1.0, additional_targets={'depth': 'image', 'mask': 'mask'})

def train_transforms_glass(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size[0] and ori_w < img_size[1]:
        transforms.append(A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size[0]), int(img_size[1]), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1., additional_targets={'depth': 'image', 'mask': 'mask'})

def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    # print("masks, labels, low_res_masks", masks.shape, labels.shape, low_res_masks.shape)
    if masks.dim() == 4:
        masks = masks.squeeze()
    if labels.dim() == 4:
        labels = labels.squeeze()
    if low_res_masks.dim() == 4:
        low_res_masks = low_res_masks.squeeze()

    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)
    # print("low_res_masks_logist_shape", low_res_masks_logist.shape)
    points, point_labels = select_random_points(masks_binary, labels, point_num = point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    # batched_input["mask_inputs"] = None
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None

    return batched_input

def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    batched_input["mask_inputs"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=False):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    # preds[preds > 0.5] = int(1)
    # preds[preds <= 0.5] = int(0)

    # mask = preds.squeeze().cpu().numpy()
    mask = preds.squeeze().cpu().detach().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: #visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size[1])
                y0_ori = int(y0 * ori_h / image_size[0])
                x1_ori = int(x1 * ori_w / image_size[1])
                y1_ori = int(y1 * ori_h / image_size[0])

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [[int((x * ori_w / image_size[1])) , int((y * ori_h / image_size[0]))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
            else:
                ori_points = [[int((x * ori_w / image_size[1])) , int((y * ori_h / image_size[0]))] for x, y in point_coords]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS , markerSize=7, thickness=2)  
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))

#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):

    # def __init__(self, weight=20.0, iou_scale=1.0):
    def __init__(self, weight=3, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()
        self.iou_loss = IOU(size_average=True).cuda()  #后加入
        self.BCE = F.binary_cross_entropy_with_logits   #后加入
        self.KLD = nn.KLDivLoss(reduction='mean', log_target=True)
    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """

        pred1, pred2 = pred
        loss1 = self.iou_loss(torch.sigmoid(pred1), mask) + self.dice_loss(pred1, mask) + self.BCE(pred1, mask)
        loss2 = self.iou_loss(torch.sigmoid(pred2), mask) + self.dice_loss(pred2, mask) + self.BCE(pred2, mask)
        loss = 3*loss1 + loss2
        return loss

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
