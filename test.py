import os
import json
import copy
import random
import logging
import argparse
import numpy as np
from importlib import import_module

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
from datasets.dataset import SAM_dataset, RandomGenerator
from utils import MLP, AverageMeter, mae

def get_dataloader_SAM(args, img_embedding_size, test_data_location):
    low_res = img_embedding_size*4
    db_test = SAM_dataset(test_data_location, transform=transforms.Compose([RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res], bbox_shift=20, get_point=3)]), inp_size=1024, type='test')
    print("The length of test set is: {}".format(len(db_test)))
    testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
    return testloader

def select(args, net_origin, name):
    net = copy.deepcopy(net_origin).cuda()
    if 'COCO' not in name.split('/')[-1]:
        net.load_lora_parameters(name, args)
    return net


def main(args, test_data, output_path, logging):  
    # ---------------Initialization---------------
    key_to_index = {}
    index_to_key = {}
    for index, key in enumerate(test_data):
        key_to_index[key]=index
        index_to_key[index]=key
    
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for name, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)
    
    pkg = import_module(f'module.{args.module}')
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam))

    model = MLP(768, class_num=len(test_data)).cuda()

    for id, key in enumerate(test_data):
        net = copy.deepcopy(net_origin).cuda()
            
        logging.info(f"------Dataset {key} is begin-------") 
        model.load_state_dict(torch.load('checkpoint/SAM1/Module_Selector.pth'))
        logging.info(f"---Validation---")

        model.eval()
        ious = AverageMeter()
        f1_scores = AverageMeter()
        mae_scores = AverageMeter()
        cnt = {}

        test_data_location = test_data[key]
        testloader = get_dataloader_SAM(args, img_embedding_size, test_data_location)
        for iter, data in enumerate(testloader):
            images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point']            
            data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)

            input_images = sam.preprocess(images)
            mid_embed, embed = sam.image_encoder(input_images, False, begin=-1, end=6)
            output = model(torch.tensor(embed).cuda())
            x=F.softmax(output, dim=1).max(dim=1)[1]
            key_cnt = x.item()
            if key_cnt not in cnt: cnt[key_cnt]=1
            else: cnt[key_cnt]+=1
            
            net = select(args, net_origin, f'checkpoint/SAM1/{index_to_key[x.item()]}.pth')
            outputs = net(mid_embed, points=points, begin=7, end=-1)
            for image_, pred_mask, gt_mask in zip(images, outputs["masks"], gt_masks):
                if len(gt_mask.size())<3:
                    gt_mask = gt_mask.unsqueeze(0)
                batch_stats = smp.metrics.get_stats(
                    torch.sigmoid(pred_mask),
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                batch_mae = mae(pred_mask, gt_mask)
                mae_scores.update(batch_mae, 1)
                ious.update(batch_iou, 1)
                f1_scores.update(batch_f1, 1)
            if logging!=None and iter%50==0:
                logging.info(
                    f'Val: [[{iter}/{len(testloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
                )
        if logging!=None:
            logging.info(
                f'Val: [[{iter}/{len(testloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
            )
            logging.info(f"Frequency of {index} task in Selecting corresponding module: {cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, 
                        default='SAMCL', help='Module (Default=SAMCL)')
    
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model (Default=vit_b)')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--img_size', type=int,
                        default=1024, help='input patch size of network input (Default=1024)')
    
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str, 
                        default="Kvasir_camo_ISTD_ISIC_cod_COCO", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod_COCO)")
    parser.add_argument('--cuda', type=int, 
                        default=-1, help='ID of GPU when using single GPU')        
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

    torch.cuda.set_device(args.cuda)
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"CUDA Index : {current_device}")
    print(f"Device Name: {device_name}")
    args.rank = 0
    print('Not using distributed mode')


    test_data = None, None
    with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    
    keys = args.order.split("_")
    shuffled_dict_test = {key: test_data[key] for key in keys}
    test_data = shuffled_dict_test

    print(f'Order is {test_data.keys()}')
    
    output_path = f'log_sam1/{args.module}_{args.order}_{seed}_testing'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(f'{output_path}/log.txt'):
        open(f'{output_path}/log.txt', 'w').close()  
    logging.basicConfig(filename=f'{output_path}/log.txt', level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(str(args))

    main(args, test_data, output_path, logging)



                
                
            
            
            
            

