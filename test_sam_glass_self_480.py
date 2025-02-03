import random
import json
import torch
import argparse
import os
import numpy as np
from metrics import SegMetrics
from tqdm import tqdm
from torch.nn import functional as F
from utils import select_random_points
from utils import FocalDiceloss_IoULoss, generate_point, save_masks, IOU
from segment_anything.build_sam_twopath import sam_model_registry
from torch.utils.data import DataLoader
from DataLoader import TestingDataset_Glass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="Please enter your work address", help="work dir")
    parser.add_argument("--run_name", type=str, default="", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=(480, 640), help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="Please enter the address of your dataset", help="train data path")
    parser.add_argument("--dataset_mode", type=str, default="test_withglass", help="select test_withoutglass or test_withglass")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'mae', 'ber'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="Please enter the address of your checkpoint", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=False, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=8, help="iter num")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=True, help="save reslut")
    parser.add_argument("--seed", type=int, default=22, help="random seed")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 10
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label' or key == 'depth':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size[0], image_size[1]),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size[0] and ori_w < image_size[1]:
        top = torch.div((image_size[0] - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size[1] - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (args.image_size[0], args.image_size[1]), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    setup_seed(args.seed)
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args).to(args.device) 

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset_Glass(data_path=args.data_path, image_size=args.image_size, mode=args.dataset_mode, requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]

        with torch.no_grad():

            depth_sup_rgb, depth_control_rgb = model.depth_encoder(batched_input["image"])     #U-shaped network with shared weights
            depth_sup_dep, depth_control_dep = model.depth_encoder(batched_input["depth"])     #U-shaped network with shared weights
            depth_sup = model.add(depth_sup_rgb, depth_sup_dep)
            depth_control = depth_control_dep + depth_control_rgb
            image_embeddings = model.image_encoder(batched_input["image"], depth_control)

            batched_input["mask_inputs"] = (torch.sigmoid(depth_sup.detach()) >= 0.5).float()
            depth_sup_rgb_binary = (torch.sigmoid(F.interpolate(depth_sup_rgb.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) > 0.5).float()
            depth_sup_dep_binary = (torch.sigmoid(F.interpolate(depth_sup_dep.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) > 0.5).float()
            points, point_labels = select_random_points(depth_sup_rgb_binary.squeeze(), depth_sup_dep_binary.squeeze(), point_num=args.point_num)
            batched_input["point_coords"], batched_input['point_labels'] = torch.as_tensor(points), torch.as_tensor(point_labels)
            batched_input = to_device(batched_input, args.device)

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_predict", args.dataset_mode)
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point-1:
                    pseudo_labels = (torch.sigmoid(F.interpolate(depth_sup.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) >= 0.5).float()
                    batched_input = generate_point(masks, pseudo_labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.cat(point_coords, dim=1)
                    batched_input["point_labels"] = torch.cat(point_labels, dim=1)

                if args.prompt_path is None:
                    prompt_dict[img_name] = {
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                    }

            points_show = (torch.cat(point_coords, dim=1), torch.cat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        IOU_Loss = IOU()
        loss = IOU_Loss(torch.sigmoid(masks), ori_labels)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir, f'{args.image_size}, {args.dataset_mode}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
