import os
import json
import copy
import random
import logging
import argparse
import numpy as np
from importlib import import_module

import torch
from torch import optim
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
from datasets.dataset import SAM_dataset, RandomGenerator
from utils import train, configure_opt, CustomDataset, MLP, train_embed, AverageMeter, mae

def get_dataloader_MLP(args, Embedding, map_dict):
    if args.cuda==-1:
        train_datasets_MLP = CustomDataset(Embedding, map_dict.copy())
        train_sampler_MLP = torch.utils.data.distributed.DistributedSampler(train_datasets_MLP)
        train_batch_sampler_MLP = torch.utils.data.BatchSampler(train_sampler_MLP, int(24/dist.get_world_size()), drop_last=True)
        if args.cuda==-1 and args.rank==0: print("The length of MLP train set is: {}".format(len(train_datasets_MLP)))
        train_dataloader_MLP = DataLoader(train_datasets_MLP, batch_sampler=train_batch_sampler_MLP, pin_memory=False)
        return train_dataloader_MLP, train_sampler_MLP
    else:
        train_datasets_MLP = CustomDataset(Embedding, map_dict.copy())
        train_dataloader_MLP = DataLoader(train_datasets_MLP, batch_size=min(len(train_datasets_MLP),24), shuffle=True, drop_last=True)
        return train_dataloader_MLP

def get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location):
    low_res = img_embedding_size*4
    if args.cuda==-1:
        db_train = SAM_dataset(train_data_location, transform=transforms.Compose([RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res], bbox_shift=20, get_point=3)]), inp_size=1024, type='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
        if args.cuda==-1 and args.rank==0: print("The length of train set is: {}".format(len(db_train)))
        trainloader = DataLoader(db_train, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True)

        db_test = SAM_dataset(test_data_location, transform=transforms.Compose([RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res], bbox_shift=20, get_point=3)]), inp_size=1024, type='test')
        test_sampler = torch.utils.data.distributed.DistributedSampler(db_test)
        test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, 1, drop_last=False)
        if args.cuda==-1 and args.rank==0: print("The length of test set is: {}".format(len(db_test)))
        testloader = DataLoader(db_test, batch_sampler=test_batch_sampler, num_workers=8, pin_memory=True)
        return trainloader, testloader, train_sampler, test_sampler
    else:
        db_train = SAM_dataset(train_data_location, transform=transforms.Compose([RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res], bbox_shift=20, get_point=3)]), inp_size=1024, type='train')
        if args.cuda==-1 and args.rank==0: print("The length of train set is: {}".format(len(db_train)))
        trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

        db_test = SAM_dataset(test_data_location, transform=transforms.Compose([RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res], bbox_shift=20, get_point=3)]), inp_size=1024, type='test')
        if args.cuda==-1 and args.rank==0: print("The length of test set is: {}".format(len(db_test)))
        testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
        return trainloader, testloader

def get_forgetting_metric(now, lenth, AIJ, logging):
    AA = [0., 0., 0.]
    FM = [0., 0., 0.]
    FT = [0., 0., 0.]
    for id in range(lenth+1):
        AA[0]+=AIJ[(now, id)][0]/(lenth+1)
        AA[1]+=AIJ[(now, id)][1]/(lenth+1)
        AA[2]+=AIJ[(now, id)][2]/(lenth+1)
        FM[0]+=(AIJ[(id, id)][0] - AIJ[(now, id)][0])/(lenth+1)
        FM[1]+=(AIJ[(id, id)][1] - AIJ[(now, id)][1])/(lenth+1)
        FM[2]+=(AIJ[(id, id)][2] - AIJ[(now, id)][2])/(lenth+1)
        if id<lenth:
            FT[0]+=AIJ[(id, id+1)][0]/lenth
            FT[1]+=AIJ[(id, id+1)][1]/lenth
            FT[2]+=AIJ[(id, id+1)][2]/lenth
            
    for i in range(len(AA)):
        AA[i] = AA[i].cpu().item()
        FM[i] = FM[i].cpu().item()
        if torch.is_tensor(FT[i]):
            FT[i] = FT[i].cpu().item()
    if args.rank==0:
        logging.info("AA: {}".format(AA))
        logging.info("FM: {}".format(FM))
        logging.info("FT: {}".format(FT))

def select(args, net_origin, name):
    net = copy.deepcopy(net_origin).cuda()
    if 'COCO' not in name.split('/')[-1]:
        net.load_lora_parameters(name, args)
    if args.cuda==-1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.rank], find_unused_parameters=True)
        net = net.module
    return net


def main(args, train_data, test_data, output_path, logging):  
    # ---------------Initialization---------------
    key_to_index = {}
    index_to_key = {}
    for index, key in enumerate(train_data):
        key_to_index[key]=index
        index_to_key[index]=key
    
    task_number = len(key_to_index)
    
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for name, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)
    
    pkg = import_module(f'module.{args.module}')
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam))
    
    model = MLP(768, class_num=task_number).cuda()
    if args.cuda==-1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)
        model = model.module
    
    AIJ = {}
    Embedding = []
    testloader, test_sampler = {}, {}
    for id, key in enumerate(train_data):
        test_data_location = test_data[key]
        train_data_location= train_data[key]
        if args.cuda==-1:
            _, testloader[id], _, test_sampler[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)
        else:
            _, testloader[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)

    # ---------------Continual Segmentation---------------
    for id, key in enumerate(train_data):
        net = copy.deepcopy(net_origin).cuda()
        if args.cuda==-1:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.rank], find_unused_parameters=True)
            net = net.module
            
        if args.cuda!=-1 or args.rank==0:
            logging.info(f"------Dataset {key} is begin-------") 
        test_data_location = test_data[key]
        train_data_location= train_data[key]
        if args.cuda==-1:
            trainloader, _, train_sampler, _ = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)
        else:
            trainloader, testloader[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)

        if args.cuda==-1 or args.rank==0: 
            logging.info(f'---Generating Embedding---')
        cnt = 0
        sam.eval()
        for data in trainloader:
            images, gt_masks = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True)
            if args.cuda==-1:
                input_images = sam.preprocess(images)
                _, embed = sam.image_encoder(input_images, False, begin=-1, end=6)
            else:
                _, embed = net(images, False, begin=-1, end=6)
            embed = torch.tensor(embed).cuda()
            if args.cuda==-1:
                embed_list = [torch.zeros_like(embed.clone().detach()) for _ in range(dist.get_world_size())]
                dist.all_gather(embed_list, embed)
                for lenth in range(len(embed_list)):
                    for index in range(len(embed_list[lenth])):
                        Embedding.append((embed_list[lenth][index], key))
                        cnt+=1
                        if cnt>=args.num_embedding:break
                    if cnt>=args.num_embedding:break
                if cnt>=args.num_embedding:break
            else:
                for index in range(len(embed)):
                    Embedding.append((embed[index], key))
                    cnt+=1
                    if cnt>=args.num_embedding:break
                if cnt>=args.num_embedding:break
        
        optimizer, scheduler = configure_opt(model=net, max_epoch=args.epoch, lr=0.005, weight_decay=None, eta_min=1e-7)
        is_distributed=None
        if args.cuda==-1:
            is_distributed=(train_sampler, test_sampler[id])
            
        if 'COCO' not in key:
            if args.cuda!=-1 or  args.rank==0: 
                logging.info(f'---Train model---')
            train(Epoch=args.epoch, model=net, optimizer=optimizer, scheduler=scheduler,\
                    train_dataloader=trainloader, test_dataloader=testloader[id], logging=logging, output_path=output_path,\
                    args=args, is_distributed=is_distributed)
            net.save_lora_parameters(f'{output_path}/{key}.pth')
        
        if args.cuda!=-1 or args.rank==0: 
            logging.info(f'---Train Module Selector---')
        if args.cuda==-1:
            train_dataloader_MLP, train_sampler_MLP = get_dataloader_MLP(args, Embedding, key_to_index)
        else:
            train_dataloader_MLP = get_dataloader_MLP(args, Embedding, key_to_index)

        optimizer = optim.SGD(params=model.parameters(), lr=0.01)
        model.train()
        for epoch in range(25):
            if args.cuda==-1:
                train_sampler_MLP.set_epoch(epoch)
            train_embed(model, optimizer, train_dataloader_MLP)
        torch.save(model.state_dict(), f'{output_path}/Module_Selector.pth')
        
        # model.load_state_dict(torch.load('Module_Selector.pth'))
        if args.cuda!=-1 or args.rank==0:
            logging.info(f"---Validation---")
        
        for index in range(min(len(train_data), id+2)):
            model.eval()
            ious = AverageMeter()
            f1_scores = AverageMeter()
            mae_scores = AverageMeter()
            cnt = {}
            if args.cuda!=-1 or args.rank==0: 
                logging.info(f'-----{index} of {index_to_key[index]} begin test-------')
            for iter, data in enumerate(testloader[index]):
                images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point']            
                data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)

                input_images = sam.preprocess(images)
                mid_embed, embed = sam.image_encoder(input_images, False, begin=-1, end=6)
                output = model(torch.tensor(embed).cuda())
                x=F.softmax(output, dim=1).max(dim=1)[1]
                if args.cuda==-1:
                    key_list = [torch.zeros_like(x.clone().detach()) for _ in range(dist.get_world_size())]
                    dist.all_gather(key_list, x)
                    for i in range(len(key_list)):
                        value = key_list[i].item()
                        if value not in cnt: cnt[value]=1
                        else: cnt[value]+=1
                else:
                    key_cnt = x.item()
                    if key_cnt not in cnt: cnt[key_cnt]=1
                    else: cnt[key_cnt]+=1
                
                net = select(args, net_origin, f'{output_path}/{index_to_key[x.item()]}.pth')
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
                    if args.cuda==-1:
                        iou_list = [torch.zeros_like(batch_iou.clone().detach()) for _ in range(dist.get_world_size())]
                        f1_list = [torch.zeros_like(batch_f1.clone().detach()) for _ in range(dist.get_world_size())]
                        mae_list = [torch.zeros_like(batch_mae.clone().detach()) for _ in range(dist.get_world_size())]
                        dist.all_gather(iou_list, batch_iou)
                        dist.all_gather(f1_list, batch_f1)
                        dist.all_gather(mae_list, batch_mae)
                        for lenth in range(len(iou_list)):
                            mae_scores.update(mae_list[lenth], 1)
                            ious.update(iou_list[lenth], 1)
                            f1_scores.update(f1_list[lenth], 1)
                    else:
                        mae_scores.update(batch_mae, 1)
                        ious.update(batch_iou, 1)
                        f1_scores.update(batch_f1, 1)
                if logging!=None and iter%50==0:
                    if args.cuda!=-1 or args.rank==0: 
                        logging.info(
                            f'Val: [[{iter}/{len(testloader[index])}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
                        )
            if logging!=None:
                if args.rank==0 or args.cuda!=-1: 
                    logging.info(
                        f'Val: [[{iter}/{len(testloader[index])}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
                    )
                    logging.info(f"Frequency of {index} task in Selecting corresponding module: {cnt}")
            AIJ[(id,index)] = (ious.avg, f1_scores.avg, mae_scores.avg)
        
        if args.rank==0 or args.cuda!=-1: 
            if 'COCO' in key:
                get_forgetting_metric(id, id-1, AIJ, logging)
            else:
                get_forgetting_metric(id, id, AIJ, logging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, 
                        default='SAMCL', help='Module (Default=SAMCL)')
    parser.add_argument('--num_embedding', type=int, 
                        default=300, help='Number of stored embedding per domain (default=300)')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu (Default=2)')
    parser.add_argument('--lr', type=float,
                        default=0.005, help='Learning rate (Default=0.005)')
    parser.add_argument('--epoch', type=int,
                        default=20, help='Epoch (Default=20)')
    
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model (Default=vit_b)')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--img_size', type=int,
                        default=1024, help='input patch size of network input (Default=1024)')
    
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str, 
                        default="Kvasir_camo_ISTD_ISIC_cod_COCO", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod)")
    parser.add_argument('--cuda', type=int, 
                        default=-1, help='ID of GPU when using single GPU (cuda=-1 means using distributed GPU)')        
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

    if args.cuda==-1 and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        master_addr = os.environ['MASTER_ADDR']  
        master_port = os.environ['MASTER_PORT']  
        print(f"rank = {args.rank} is initialized in {master_addr}:{master_port}; local_rank = {args.gpu}")
        torch.cuda.set_device(args.gpu)
        args.dist_url = 'env://'  
        args.dist_backend = 'nccl' 
        print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,                      
                                world_size=args.world_size, rank=args.rank)
        dist.barrier() 
    
    else:
        torch.cuda.set_device(args.cuda)
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA Index : {current_device}")
        print(f"Device Name: {device_name}")
        args.rank = 0
        print('Not using distributed mode')


    train_data, test_data = None, None
    with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    keys = args.order.split("_")
    shuffled_dict_train = {key: train_data[key] for key in keys}
    shuffled_dict_test = {key: test_data[key] for key in keys}
    train_data = shuffled_dict_train
    test_data = shuffled_dict_test

    if args.rank==0:
        print(f'Order is {train_data.keys()}')
    
    output_path = f'log_sam1/{args.module}_{args.order}_{args.num_embedding}_{seed}_training'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(f'{output_path}/log.txt'):
        open(f'{output_path}/log.txt', 'w').close()  
    logging.basicConfig(filename=f'{output_path}/log.txt', level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(str(args))

    main(args, train_data, test_data, output_path, logging)



                
                
            
            
            
            

