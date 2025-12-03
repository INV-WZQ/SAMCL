import cv2
import numpy as np
import torchvision
from tqdm import tqdm
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks

ALPHA = 0.8
GAMMA = 2

class CustomDataset(Dataset):
    def __init__(self, data, map_dict):
        self.data = data
        self.map_dict = map_dict.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (torch.tensor(self.data[idx][0], dtype=torch.float32), \
            torch.tensor(self.map_dict[self.data[idx][1]], dtype=torch.int64))
        return sample

class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        class_num=1
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, embedding_dim//4)
        self.norm2 = nn.BatchNorm1d(embedding_dim//4)
        self.layer3 = nn.Linear(embedding_dim//4, embedding_dim//4)
        self.norm3 = nn.BatchNorm1d(embedding_dim//4)
        self.layer4 = nn.Linear(embedding_dim//4, class_num)
        self.out_cnt = class_num
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(self.layer1(x))
        out = F.relu(self.norm2(self.layer2(out)))
        out = F.relu(self.norm3(self.layer3(out)))
        out = self.layer4(out)
        return out
    
    def set_classify(self, cnt=1):
        new_layer = nn.Linear(self.embedding_dim // 4, self.out_cnt+cnt)  # 新的层
        new_layer.weight.data[:self.out_cnt, :] = self.layer4.weight.data # 复制权重
        new_layer.bias.data[:self.out_cnt] = self.layer4.bias.data  
        self.out_cnt+=cnt
        self.layer4 = new_layer
    

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
        focal_loss = focal_loss.mean()

        return focal_loss

class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

def train_embed(model, optimizer, data_loader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def test_embed(model, data_loader, logging=None):
    model.eval()
    correct = 0
    for index, (input, target) in enumerate(data_loader):
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct / len(data_loader.dataset)

def configure_opt(model, max_epoch, lr=2e-4, weight_decay=None, eta_min=1e-7):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if weight_decay!=None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=eta_min)
    return optimizer, scheduler

def train(Epoch, model, optimizer, scheduler, train_dataloader, test_dataloader, logging=None, output_path=None, args=None, is_distributed=None, prev_model=None):
    model.train()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    loss_list = []
    acc_list = []
    # validate(model, test_dataloader, logging, args=args, is_distributed=is_distributed) 
    for epoch in tqdm(range(1, Epoch+1)):
        if is_distributed is not None:
            is_distributed[0].set_epoch(epoch)
            is_distributed[1].set_epoch(epoch)
        cnt = 0
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        for ite, data in enumerate(train_dataloader):
            images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point'] 
            data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)
            outputs = model(images, points=points)
            pred_masks, iou_predictions = outputs['masks'], outputs['iou_predictions'] 
            
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)

            loss_focal = torch.tensor(0.).cuda()
            loss_dice = torch.tensor(0.).cuda()
            loss_iou = torch.tensor(0.).cuda()
            for image_, pred_mask, gt_mask, iou_prediction in zip(images, pred_masks, gt_masks, iou_predictions):
                if len(gt_mask.size())<3:
                    gt_mask = gt_mask.unsqueeze(0)
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask.float())
                loss_dice += dice_loss(pred_mask, gt_mask)
                if batch_iou.dim() == 2:
                    batch_iou = batch_iou.squeeze(0)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
            
            loss_total = loss_focal + 10 * loss_dice + loss_iou 
            if 'EWC' in args.module:
                
                loss_total += model.penalty()
            if 'ER' in args.module:
                loss_total += model.forward_replay(args)
                model.add_data(images, gt_masks, points[0], points[1])
            if 'O_LoRA' in args.module:
                loss_total += model.forward_(args)
            if 'LAG' in args.module and prev_model is not None:
                loss_total += model.forward_(args, images, points, outputs, optimizer, scheduler, prev_model)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if is_distributed is not None:
                iou_list = [torch.zeros_like(loss_iou.clone().detach()) for _ in range(dist.get_world_size())]
                focal_list = [torch.zeros_like(loss_focal.clone().detach()) for _ in range(dist.get_world_size())]
                dice_list = [torch.zeros_like(loss_dice.clone().detach()) for _ in range(dist.get_world_size())]
                total_list = [torch.zeros_like(loss_total.clone().detach()) for _ in range(dist.get_world_size())]

                dist.all_gather(iou_list, loss_iou)
                dist.all_gather(focal_list, loss_focal)
                dist.all_gather(dice_list, loss_dice)
                dist.all_gather(total_list, loss_total)

                for lenth in range(len(iou_list)):
                    focal_losses.update(focal_list[lenth].item(), 1)
                    dice_losses.update(dice_list[lenth].item(), 1)
                    iou_losses.update(iou_list[lenth].item(), 1)
                    total_losses.update(total_list[lenth].item(), 1)
            else:
                focal_losses.update(loss_focal.item(), 1)
                dice_losses.update(loss_dice.item(), 1)
                iou_losses.update(loss_iou.item(), 1)
                total_losses.update(loss_total.item(), 1)
                
            if logging!=None and ite%10==0:
                if  args.cuda!=-1 or args.rank==0:
                    logging.info(f'ID: {args.rank} | Epoch: [{epoch}][{ite+1}/{len(train_dataloader)}]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
        
        if args.cuda!=-1 or args.rank==0:
            loss_list.append(total_losses.avg)
            
        if logging!=None:
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'ID: {args.rank}') 
                logging.info(f'Epoch: [{epoch}][{ite+1}/{len(train_dataloader)}]'
                        f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                        f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                        f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                        f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
        if epoch%5==0:
            validate(model, test_dataloader, logging, args=args, is_distributed=is_distributed) 

@torch.no_grad()
def validate(model, test_loader, logging=None, args=None, is_distributed=None):
    model.eval()
    cnt = 0
    ious_ = AverageMeter() 
    ious = AverageMeter()
    f1_scores = AverageMeter()
    mae_scores = AverageMeter()
    ious_list = []
    f1_list = []
    for iter, data in enumerate(test_loader):
        images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point']
        data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)
        outputs = model(images, points=points)
        for image_, pred_mask, gt_mask in zip(images, outputs["masks"], gt_masks):
            if len(gt_mask.size())<3:
                gt_mask = gt_mask.unsqueeze(0)
            batch_stats = smp.metrics.get_stats(
                torch.sigmoid(pred_mask),
                gt_mask.int(),
                mode='binary',
                threshold=0.5,
            )
            batch_iou_ = calc_iou(pred_mask, gt_mask)
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            batch_mae = mae(pred_mask, gt_mask)
            if is_distributed is not None:
                iou_list_ = [torch.zeros_like(batch_iou_.clone().detach()) for _ in range(dist.get_world_size())]
                iou_list = [torch.zeros_like(batch_iou.clone().detach()) for _ in range(dist.get_world_size())]
                f1_list = [torch.zeros_like(batch_f1.clone().detach()) for _ in range(dist.get_world_size())]
                mae_list = [torch.zeros_like(batch_mae.clone().detach()) for _ in range(dist.get_world_size())]
                dist.all_gather(iou_list_, batch_iou_)
                dist.all_gather(iou_list, batch_iou)
                dist.all_gather(f1_list, batch_f1)
                dist.all_gather(mae_list, batch_mae)
                for lenth in range(len(iou_list)):
                    ious_.update(iou_list_[lenth], 1)
                    mae_scores.update(mae_list[lenth], 1)
                    ious.update(iou_list[lenth], 1)
                    f1_scores.update(f1_list[lenth], 1)
            else:
                mae_scores.update(batch_mae, 1)
                ious.update(batch_iou, 1)
                f1_scores.update(batch_f1, 1)
        
        if logging!=None and iter%50==0:
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'ID: {args.rank}') 
                logging.info(
                    f'Val: [[{iter}/{len(test_loader)}]: Mean IoU: [{ious.avg.item():.4f}] -- Mean F1: [{f1_scores.avg.item():.4f}] -- MAE: [{mae_scores.avg.item():.4f}]'
                )
    if logging!=None:
        if args.cuda!=-1 or args.rank==0:
            logging.info(f'ID: {args.rank}') 
            logging.info(
            f'Val: [[{iter}/{len(test_loader)}]: Mean IoU: [{ious.avg.item():.4f}] -- Mean F1: [{f1_scores.avg.item():.4f}] -- MAE: [{mae_scores.avg.item():.4f}]'
            )
    return ious.avg, f1_scores.avg, mae_scores.avg

def generate_heatmap(sigma, pos=None, h=1024, w=1024):
    ans = None
    for batch_id in range(pos[0].size(0)):    
        img = np.zeros((h, w))
        for id in range(len(pos[0][batch_id])):
            x, y = pos[0][batch_id][id]
            x, y = int(x.int().cpu().item()*h/1024), int(y.int().cpu().item()*w/1024)
            img[x][y] = 1
        heatmap = cv2.GaussianBlur(img, sigma, 0)
        am = np.amax(img)
        tmp = torch.tensor(heatmap*255).cuda()
        tmp = tmp.unsqueeze(0).unsqueeze(0) 
        # tmp = F.interpolate(tmp, size=(h, w), mode='bilinear', align_corners=False)
        if ans==None:
            ans = tmp
        else:
            ans = torch.cat((ans, tmp), dim=0)
    return ans

def draw_image(image, masks, boxes, labels, points=None, alpha=0.4):
    if points is not None:
        points = points[0][0]
    if masks is not None:
        masks = (masks.cpu()>=0.5).bool()
        # masks = masks.bool()
        if len(masks.size())>3:
            masks = masks[0]
    image = image.cpu()
    image = image.byte()
    if len(image.size())>3:
        image = image[0]
        if boxes is not None:
            boxes = boxes[0].unsqueeze(0)
    if boxes is not None:
        if image.max()<=1:
            image = (image * 255)
        image = torchvision.utils.draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=None, width=2)

    if points is not None:
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.uint8)
        for point in range(points.size(0)):
            now = points[point].numpy()  
            x, y = int(now[0]), int(now[1])  
            image = np.ascontiguousarray(image)
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        return image
    
    if masks is not None:
        if len(masks.size())>3:
            masks = masks.squeeze(0)
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

def mse(imageA, imageB):
    # NOTE: the two images must have the same dimension
    imageA = (imageA>0)
    imageB = (imageB>0)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) 
    err /= float(imageA.shape[0] * imageA.shape[1]) 
    return err

def mae(pred, gt):
    # NOTE: the two images must have the same dimension
    pred = torch.sigmoid(pred)>=0.5
    pred = pred.int()
    if pred.is_cuda:
        pred = pred.cpu()
    if gt.is_cuda:
        gt = gt.cpu()

    err = np.sum(np.abs(pred.numpy().astype("float") - gt.numpy().astype("float")))
    err /= float(pred.shape[1] * pred.shape[2]) 
    return torch.tensor(err).cuda()

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
