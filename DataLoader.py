import os
import json
import random
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, train_transforms_aug, get_boxes_from_mask, init_point_sampling, train_transforms_glass, get_boxes_from_mask_glass, init_point_sampling_glass
from PIL import Image, ImageFile
from torchvision import transforms

# from utils import Resize
class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num
        dataset_rgb = json.load(open(os.path.join(data_path, f'label2image_{mode}.json'), "r"))
        self.image_paths = list(dataset_rgb.values())
        self.label_paths = list(dataset_rgb.keys())
        dataset_depth = json.load(open(os.path.join(data_path, f'label2depth_{mode}.json'), "r"))
        self.depth_paths = list(dataset_depth.values())
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.to_tensor = transforms.ToTensor()
        self.to_copy = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        try:
            depth = Image.open(self.depth_paths[index])
            depth = np.array(depth).astype(np.uint8)
            depth = self.to_tensor(depth)
            if depth.shape[2]==1:
                depth = self.to_copy(depth)
            depth = np.array(depth, dtype=np.float64).transpose((1, 2, 0))
        except:
            print(self.depth_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)
        if not isinstance(ori_np_mask, np.ndarray):
            ori_np_mask = np.array(ori_np_mask)

        augments = transforms(image=image, depth=depth, mask=ori_np_mask)
        image, depth, mask = augments['image'], augments['depth'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, box_num=1, max_pixel = 0)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        image_input["depth"] = depth

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)


class TestingDataset_Glass(Dataset):

    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True,
                 prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        dataset_rgb = json.load(open(os.path.join(data_path, f'label2image_{mode}.json'), "r"))
        self.image_paths = list(dataset_rgb.values())
        self.label_paths = list(dataset_rgb.keys())
        dataset_depth = json.load(open(os.path.join(data_path, f'label2thermal_{mode}.json'), "r"))
        self.depth_paths = list(dataset_depth.values())
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.to_tensor = transforms.ToTensor()
        self.to_copy = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print("报错图片", self.image_paths[index])

        try:
            depth = Image.open(self.depth_paths[index])
            depth = np.array(depth).astype(np.uint8)
            depth = self.to_tensor(depth)
            depth = self.to_copy(depth)
            depth = np.array(depth, dtype=np.float64).transpose((1, 2, 0))

        except:
            print("报错图片", self.depth_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)

        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(
            bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms_glass(self.image_size, h, w)

        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if ori_np_mask.dtype != np.float32:
            ori_np_mask = ori_np_mask.astype(np.float32)

        augments = transforms(image=image, depth=depth, mask=ori_np_mask)
        image, depth, mask = augments['image'], augments['depth'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask_glass(mask, max_pixel=0)
            point_coords, point_labels = init_point_sampling_glass(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        image_input["depth"] = depth

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask

        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset_rgb = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset_rgb.keys())
        self.label_paths = list(dataset_rgb.values())
        dataset_depth = json.load(open(os.path.join(data_dir, f'depth2label_{mode}.json'), "r"))
        self.depth_paths = list(dataset_depth.keys())
        self.to_tensor = transforms.ToTensor()
        self.to_copy = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        try:
            depth = Image.open(self.depth_paths[index])
            depth = np.array(depth).astype(np.uint8)
            depth = self.to_tensor(depth)
            if depth.shape[2]==1:
                depth = self.to_copy(depth)
            depth = np.array(depth, dtype=np.float64).transpose((1, 2, 0))

        except:
            print("报错图片", self.depth_paths[index])

        h, w, _ = image.shape
        transforms_fu = train_transforms_aug(self.image_size, h, w)  #数据增强

        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []

        mask_path = self.label_paths[index]
        pre_mask = cv2.imread(mask_path, 0)

        if pre_mask.max() == 255:
            pre_mask = pre_mask / 255

            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            if pre_mask.dtype != np.float32:
                pre_mask = pre_mask.astype(np.float32)

            augments = transforms_fu(image=image, depth=depth, mask=pre_mask)
            image_tensor, depth_tensor, mask_tensor = augments['image'], augments['depth'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)
        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_input["depth"] = depth_tensor.unsqueeze(0)

        image_name = self.image_paths[index].split('/')[-1]
        # depth_name = self.depth_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

class TrainingDataset_Glass(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset_rgb = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset_rgb.keys())
        self.label_paths = list(dataset_rgb.values())
        dataset_depth = json.load(open(os.path.join(data_dir, f'thermal2label_{mode}.json'), "r"))
        self.depth_paths = list(dataset_depth.keys())
        self.to_tensor = transforms.ToTensor()
        self.to_copy = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        try:
            depth = Image.open(self.depth_paths[index])
            depth = np.array(depth).astype(np.uint8)
            depth = self.to_tensor(depth)
            depth = self.to_copy(depth)
            depth = np.array(depth, dtype=np.float64).transpose((1, 2, 0))

        except:
            print("报错图片", self.depth_paths[index])


        h, w, _ = image.shape
        transforms_fu = train_transforms_aug(self.image_size, h, w)

        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = self.label_paths[index]
        pre_mask = cv2.imread(mask_path, 0)

        if pre_mask.max() == 255:
            pre_mask = pre_mask / 255
        else:
            pass

        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if pre_mask.dtype != np.float32:
            pre_mask = pre_mask.astype(np.float32)

        augments = transforms_fu(image=image, depth=depth, mask=pre_mask)
        image_tensor, depth_tensor, mask_tensor = augments['image'], augments['depth'], augments['mask'].to(torch.int64)

        boxes = get_boxes_from_mask_glass(mask_tensor)
        point_coords, point_label = init_point_sampling_glass(mask_tensor, self.point_num)
        # print("mask_tensor", mask_tensor)
        masks_list.append(mask_tensor)
        boxes_list.append(boxes)
        point_coords_list.append(point_coords)
        point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_input["depth"] = depth_tensor.unsqueeze(0)

        image_name = self.image_paths[index].split('/')[-1]

        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    train_dataset = TrainingDataset("/root/autodl-tmp/SAM_Med2D_main/dataset/", image_size=224, mode='train', requires_name=True, point_num=1, mask_num=1)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["depth"].shape, batched_image["label"].shape)

    # test_dataset = TestingDataset("/root/autodl-tmp/SAM_Med2D_main/dataset/", image_size=224, mode='test', requires_name=True, point_num=1)
    # print("Dataset:", len(test_dataset))
    # test_batch_sampler = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # for i, batched_image in enumerate(tqdm(test_batch_sampler)):
    #     # batched_image = stack_dict_batched(batched_image)
    #     print(batched_image["image"].shape, batched_image["depth"].shape, batched_image["label"].shape)

