import time
import torch
import shutil
import argparse
import random
import os
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from segment_anything.build_sam_twopath import sam_model_registry
from metrics import SegMetrics
from segment_anything.optim.Ranger import Ranger
from tqdm import tqdm
from torch.nn import functional as F
from backbone.resnet import Backbone_ResNet50_in3
from utils import select_random_points

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
    parser.add_argument("--run_name", type=str, default="xxmodel", help="run model name")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--image_size", type=int, default=(416, 416), help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="Please enter the address of your dataset", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'mae', 'ber'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="/home/UniSAM/segment_anything/pretrain_model/sam_vit_b_01ec64.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=True, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    parser.add_argument("--dataset_name", type=str, default='mirror', help="dataset_name")

    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
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

def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
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

    masks = F.interpolate(low_res_masks,(args.image_size[0], args.image_size[1]), mode="bilinear", align_corners=False,)

    return masks, low_res_masks, iou_predictions

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, args.device)
            
            if random.random() > 0.5:
                batched_input["point_coords"] = None
                flag = "boxes"
            else:
                batched_input["boxes"] = None
                flag = "point"

            for n, value in model.image_encoder.named_parameters():
                if "Adapter" in n:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

            if args.use_amp:
                labels = batched_input["label"].half()
                depth_embeddings, depth_sup, depth_control = model.depth_encoder(batched_input["depth"])
                image_embeddings = model.image_encoder(batched_input["image"].half())
      
                batch, _, _, _ = image_embeddings.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = image_embeddings[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)

                batched_input["mask_inputs"] = (torch.sigmoid(depth_sup) >= 0.5).float()
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
                loss = criterion((masks, depth_embeddings, F.interpolate(depth_sup, size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)), labels, iou_predictions)

            else:
                labels = batched_input["label"]
                depth_sup_rgb, depth_control_rgb = model.depth_encoder(batched_input["image"])     #U-shaped network with shared weights
                depth_sup_dep, depth_control_dep = model.depth_encoder(batched_input["depth"])     #U-shaped network with shared weights
                depth_sup = model.add(depth_sup_rgb, depth_sup_dep)
                depth_control = depth_control_dep + depth_control_rgb

                image_embeddings = model.image_encoder(batched_input["image"], depth_control)

                batch, _, _, _ = image_embeddings.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = image_embeddings[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)
                image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
                batched_input["mask_inputs"] = (torch.sigmoid(depth_sup.detach())>=0.5).float()
                point_num = random.choice(args.point_list)
                depth_sup_rgb_binary = (torch.sigmoid(F.interpolate(depth_sup_rgb.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) > 0.5).float()
                depth_sup_dep_binary = (torch.sigmoid(F.interpolate(depth_sup_dep.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) > 0.5).float()
                points, point_labels = select_random_points(depth_sup_rgb_binary.squeeze(), depth_sup_dep_binary.squeeze(), point_num=point_num)

                batched_input["point_coords"], batched_input['point_labels'] = torch.as_tensor(points), torch.as_tensor(point_labels)
                batched_input["boxes"] = None
                batched_input = to_device(batched_input, args.device)

                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
                loss = criterion((masks, F.interpolate(depth_sup, size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)), labels, iou_predictions)
                loss.backward(retain_graph=False)

            optimizer.step()
            optimizer.zero_grad()

            if int(batch+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

            point_num = random.choice(args.point_list)
            pseudo_labels = torch.sigmoid(F.interpolate(depth_sup.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) >= 0.5
            batched_input = generate_point(masks, pseudo_labels, low_res_masks, batched_input, point_num)
            batched_input = to_device(batched_input, args.device)

            image_embeddings = image_embeddings.detach().clone()
            depth_sup = depth_sup.detach().clone()
            for n, value in model.named_parameters():
                if "image_encoder" in n:
                    value.requires_grad = False
                else:
                    value.requires_grad = True

            init_mask_num = np.random.randint(1, args.iter_point - 1)
            for iter in range(args.iter_point):
                if iter == init_mask_num or iter == args.iter_point - 1:
                    batched_input = setting_prompt_none(batched_input)

                if args.use_amp:
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                    loss = criterion(masks, labels, iou_predictions)

                else:
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                    loss = criterion((masks, F.interpolate(depth_sup, size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)), labels, iou_predictions)
                    loss.backward(retain_graph=True)
                    
                optimizer.step()
                optimizer.zero_grad()
              
                if iter != args.iter_point - 1:
                    point_num = random.choice(args.point_list)
                    pseudo_labels = (torch.sigmoid(F.interpolate(depth_sup.detach(), size=(args.image_size[0], args.image_size[1]), mode='bilinear', align_corners=False)) >= 0.5).float()
                    batched_input = generate_point(masks, pseudo_labels, low_res_masks, batched_input, point_num)
                    batched_input = to_device(batched_input, args.device)
           
                if int(batch+1) % 50 == 0:
                    if iter == init_mask_num or iter == args.iter_point - 1:
                        print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                    else:
                        print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

            if int(batch+1) % 200 == 0:
                print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
                save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
                state = {'model': model.state_dict(), 'optimizer': optimizer}
                torch.save(state, save_path)

            train_losses.append(loss.item())

            gpu_info = {}
            gpu_info['gpu_name'] = args.device 
            train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

            train_batch_metrics = SegMetrics(masks, labels, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics

def main(args):
    setup_seed(22)
    model = sam_model_registry[args.model_type](args).to(args.device)
    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=args.lr, weight_decay=1e-3)
    criterion = FocalDiceloss_IoULoss()
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / args.epochs) ** 0.9)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    # if args.use_amp:
    #     # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #     print("*******Mixed precision with Apex")
    # else:
    #     print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))

    current_time = time.strftime("%Y-%m-%d-%H-%M")
    logdir_name = f"{current_time}-({args.dataset_name}-{args.run_name})"
    logdir = os.path.join(args.work_dir, logdir_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    loggers = get_logger(logdir)
    loggers.info(f'Conf | use logdir {logdir}')

    best_loss = 1e10
    l = len(train_loader)
    best_iou = 0.1

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr:.8f}, Train: {average_loss:.4f}, metrics: {train_metrics}")

        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(logdir, f"epoch{epoch + 1}_sam.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)
        #     if args.use_amp:
        #         model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


