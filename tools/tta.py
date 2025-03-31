from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.rotnet_utils import rotate_batch
import utils.tent_shot as tent_shot_utils
import utils.t3a as t3a_utils
from utils.misc import *
import pytorch3d
from torchvision import transforms
from datasets import data_transforms
from collections import OrderedDict 
import time



############### Visualization
import os
import copy
import torch
import argparse
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from glob import glob
# from models_vis.tent import weight_average  
import json
from types import SimpleNamespace
from pytorch3d.ops import knn_points
from utils import misc
from timm.models.layers import DropPath, trunc_normal_



level = [5] 


train_transforms_rotation = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
    ]
)

train_transforms_ScaleAndTranslate = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

train_transforms_PointcloudJitter = transforms.Compose(
    [
        data_transforms.PointcloudJitter(),
    ]
)

train_transforms_PointcloudScale = transforms.Compose(
    [
        data_transforms.PointcloudScale(),
    ]
)

train_transforms_RandomHorizontalFlip = transforms.Compose(
    [
        data_transforms.RandomHorizontalFlip(),
    ]
)




def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = os.path.join(root, f'{args.dataset_name}_c')

        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'partnet':
        if args.corruption != 'clean':
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}_{args.severity}')
        else:
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}')

        inference_dataset = tta_datasets.PartNormalDataset(root=root, npoints=config.npoints, split='test',
                                                           normal_channel=config.normal, debug=args.debug)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        
    elif args.dataset_name == 'scanobject_cvpr':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)    

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for {args.tta} is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader


def load_base_model(args, config, logger, load_part_seg=False):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts, load_part_seg)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
            args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)   
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def eval_source(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject' or dataset_name == 'scanobject_cvpr':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError



    method = config.model.transformer_config.method
    episodic = config.model.transformer_config.reset   
    episodic_2 = config.model.transformer_config.reset_2
    n_aug = config.model.transformer_config.N_Aug   
    n_aug_2 = config.model.transformer_config.N_Aug_2   
    type_aug = config.model.transformer_config.Type_Aug  
    iteration = config.model.transformer_config.iteration   
    original = config.model.transformer_config.original   
    N_downsample = config.model.transformer_config.N_downsample
    Number_downsample = config.model.transformer_config.Number_downsample
    model_name = config.model.NAME
    cross_entropy = config.model.transformer_config.cross_entropy


    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path= args.experiment_path + "/")  # for saving results for easy copying to google sheet   
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name in ['scanobject', 'scanobject_wbg', 'scanobject_nbg', "scanobject_cvpr"]:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    if (model_name == "Point_MAE"):   
                        logits = base_model.module.classification_only(points, only_unmasked=False)
                    if (model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"): 
                        logits = base_model.module.forward(points)      
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:', os.path.join('source_only_results/', f'{logtime}_results.txt'))


def eval_source_rotnet(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results_rotnet/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'scanobject':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:',
                          os.path.join('source_only_results_rotnet/', f'{logtime}_results.txt'))


def weight_average(all_weights):
    """
    Compute the average of the weights from multiple models.

    Args:
        all_weights: List of state dictionaries from different models.

    Returns:
        avg_state_dict: Averaged state dictionary.
    """
    K = len(all_weights)
    avg_state_dict = OrderedDict()
    for param_name, param in all_weights[0].items():
        avg_param = sum(sd[param_name] for sd in all_weights) / K
        avg_state_dict[param_name] = avg_param
    return avg_state_dict 


def tta_rotnet(args, config, train_writer=None):
    dataset_name = args.dataset_name

    assert dataset_name is not None
    assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    # args.batch_size_tta = 48  
    args.batch_size_tta = 2      
    # args.batch_size = 1
    args.disable_bn_adaptation = True

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    
    model_name = config.model.NAME    

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                #f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_rotnet_results/')
                f_write, logtime = get_writer_to_all_result(args, config, custom_path=  args.experiment_path + "/")     
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []  
            test_label = []

            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'scanobject':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # making a batch
                    points = [points for _ in range(args.batch_size_tta)]
                    points = torch.squeeze(torch.vstack(points))   
                    pts_rot, label_rot = rotate_batch(points)
                    pts_rot, label_rot = pts_rot.cuda(), label_rot.cuda()
                    if (model_name == "Point_MAE_rotnet"):  
                        loss = base_model(0, pts_rot, 0, label_rot, tta=True)  # get out only rotnet loss
                    elif(model_name == "PointNet_rotnet" or model_name == "DGCNN_cls_rotnet" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                        loss = base_model.module.forward_rotnet(0, pts_rot, 0, label_rot, tta=True)  # get out only rotnet loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()  

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Rot Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)
                if (model_name == "Point_MAE_rotnet"): 
                    logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                elif(model_name == "PointNet_rotnet" or model_name == "DGCNN_cls_rotnet" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                    logits = base_model.module.forward(points, 0, 0, 0, tta=True)   
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    # intermediate results
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('tta_rotnet_results/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def reset(args, config, logger):
    base_model = load_base_model(args, config, logger) 
    adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)    
    

def tta_tent(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject' or dataset_name == 'scanobject_cvpr':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError
    
    layer_norm = config.model.transformer_config.layer_norm
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    npoints_ours = config.npoints_ours
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger) 
    if (layer_norm == False):
        adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)                                               
    else:
        adapted_model, optimizer = tent_shot_utils.setup_tent_shot_both(args, model=base_model)    
    #adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model, config= config)                               ### Added by Ali
    
    if dataset_name == 'modelnet' or dataset_name == 'scanobject' or dataset_name == 'scanobject_cvpr':     
        args.severity = 5
    if dataset_name == 'shapenetcore':       
        args.severity = 5 
        
    f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_tent_results/')
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    
    method = config.model.transformer_config.method
    episodic = config.model.transformer_config.reset   
    episodic_2 = config.model.transformer_config.parallel_mode
    n_aug = config.model.transformer_config.N_Aug   
    type_aug = config.model.transformer_config.Type_Aug  
    iteration = config.model.transformer_config.iteration   
    original = config.model.transformer_config.original   
    model_name = config.model.NAME
    cross_entropy = config.model.transformer_config.cross_entropy
     
    acc_final = 0
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            print("###########################################################################################", idx) 
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            
            ####################################### Data
            if (method == "Tent" or method == "Tent_Modified"):
                
                print(points.shape)
                if (dataset_name == "modelnet"):
                    points = points
                if (dataset_name == "shapenetcore" or dataset_name == "scanobject" or dataset_name == "scanobject_cvpr"):    
                    points = misc.fps(points, npoints)
                

            elif (method == "Tent_WA_FPS" or method == "Ensemble"):  
                points_list = []  
                #points_org = misc.fps(points, npoints)   
                    
                if (model_name == "Point_MAE"):              
                    for i in range(n_aug):  
                        points_org = points 
                        points = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0]     
                        points_list.append(points[None])  
                        
                    points = torch.cat((points_list), 0)             
                    
                if (model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                    for i in range(n_aug):  
                        # mean_time= 0
                        points_fps = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0]
                        # for i in range(100):
                        #     start_time = time.time()   
                        #     points_fps = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0] 
                        #     end_time = time.time()
                        #     mean_time += (end_time - start_time)
                        # print(f"################################# Execution time: {mean_time/100.} seconds")   
                        points_list.append(points_fps[None])  
                            
                    #if (dataset_name == "shapenetcore" or dataset_name == "scanobject"): 
                    points_org =  points_fps
                    points = torch.cat((points_list), 0)    
             
    
            elif (method == "Tent_WA_Aug"):   
                points_list = []  
                for i in range(n_aug):
                    #points_org = points  
                    if (type_aug == "rotation"):
                        points_ = train_transforms_rotation(points)
                        points_list.append(points_[None]) 
                        
                    elif (type_aug == "scale_transform"):
                        points_ = train_transforms_ScaleAndTranslate(points)
                        points_list.append(points_[None])
                        
                    elif (type_aug == "jitter"):
                        points_ = train_transforms_PointcloudJitter(points)    
                        points_list.append(points_[None])
                        
                    elif (type_aug == "scale"):
                        points_ = train_transforms_PointcloudScale(points)    
                        points_list.append(points_[None]) 
                        
                    elif (type_aug == "h_flip"):
                        points_ = train_transforms_RandomHorizontalFlip(points)           
                        points_list.append(points_[None])    
                        
                points_org = points_             
                points = torch.cat((points_list), 0)  
                
                
            elif (method == "Tent_WA_Aug_FPS"):   
                points_list = []  
                if (model_name == "Point_MAE"):   
                    for i in range(n_aug):
                        
                        if (type_aug == "jitter"):
                            points = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0]
                            points = train_transforms_PointcloudJitter(points) 
                            points_list.append(points[None])
                    
                points_org = points           
                points = torch.cat((points_list), 0)                
                
            
            ####################################### Network  
            if (method == "Tent"):
                logits = tent_shot_utils.forward_and_adapt_tent(points, adapted_model, optimizer)   
                
            elif (method == "Tent_Modified"):     
                all_weights = []
                if episodic:
                    reset(args, config, logger)        
                adapted_model.train() 
                if (model_name == "Point_MAE"):   
                    if (cross_entropy == False):
                        logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_modified(points, adapted_model, optimizer, all_weights, iteration, layer_norm)
                    else:
                        logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_modified_with_CE(points, adapted_model, optimizer, all_weights, iteration, layer_norm)    
                elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                    if (cross_entropy == False):
                        logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA(points, adapted_model, optimizer, all_weights, iteration, layer_norm)
                    else:
                        logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA_with_CE(points, adapted_model, optimizer, all_weights, iteration, layer_norm) 
                            
                avg_state_dict = weight_average(all_weights_cmp)
                adapted_model.load_state_dict(avg_state_dict, strict=False)
                adapted_model.eval()
                if (model_name == "Point_MAE"):     
                    logits = adapted_model.module.classification_only(points, only_unmasked=False)  
                elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                    logits = adapted_model.forward(points)      
                         
                
            elif (method == "Tent_WA_FPS" or method == "Tent_WA_Aug" or method == "Tent_WA_Aug_FPS"):   
                if episodic:
                    reset(args, config, logger)   
                adapted_model.train() 
                all_weights = []      
                for i in range(n_aug):            
                #for i in range(n_aug + Number_downsample):       
                    if (model_name == "Point_MAE"):    
                        if (cross_entropy == False):
                            # mean_time= 0
                            # mean_time_2= 0
                            logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)
                
                            # for f in range(100):
                            #     start_time = time.time()
                            #     logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)
                            #     end_time = time.time()
                            #     mean_time += (end_time - start_time)
                            # for f in range(100):    
                            #     start_time = time.time()    
                            #     avg_state_dict = weight_average(all_weights_cmp[:6]) 
                            #     end_time = time.time()
                            #     mean_time_2 += (end_time - start_time)

                            # print(f"Execution time: {mean_time/100.} seconds")
                            # print(f"Execution time_2: {mean_time_2/100.} seconds")

                            if (episodic_2):
                                reset(args, config, logger)   
                        else:
                            logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_with_WA_with_CE(points[i], adapted_model, optimizer, all_weights, iteration)
                            if (episodic_2): 
                                reset(args, config, logger)    
                                
                    elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                        if (cross_entropy == False):
                            logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)
                            """mean_time= 0
                            for f in range(100):
                                start_time = time.time()
                                logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)  
                                end_time = time.time()
                                mean_time += (end_time - start_time)
                            print(f"Execution time: {mean_time/100.} seconds")  """
                        
                            if (episodic_2):   
                                reset(args, config, logger)  
                                
                        else:
                            logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA_with_CE(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)        
                            if (episodic_2):   
                                reset(args, config, logger)  
                                
                avg_state_dict = weight_average(all_weights_cmp)
                adapted_model.load_state_dict(avg_state_dict, strict=False) 
                adapted_model.eval() 
                if (original):
                    if (model_name == "Point_MAE"):
                        logits = adapted_model.module.classification_only(points_org, only_unmasked=False)       
                    elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                        logits = adapted_model.forward(points_org)    
                else:                   
                    logits = adapted_model.module.classification_only(points[0], only_unmasked=False)                
                     
                     
            elif (method == "Ensemble"):   
                if episodic:
                    reset(args, config, logger)   
                adapted_model.train() 
                logits_list = []      
                for i in range(n_aug):         
                #for i in range(n_aug + Number_downsample):       
                    if (model_name == "Point_MAE"):    
                        if (cross_entropy == False):
                            logits = tent_shot_utils.forward_and_adapt_tent_with_WA_Ensemble(points[i], adapted_model, optimizer, iteration, layer_norm)
                        else:
                            logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_with_WA_with_CE(points[i], adapted_model, optimizer, all_weights, iteration)
                            if (episodic_2): 
                                reset(args, config, logger) 
                                
                    elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                        logits, all_weights_cmp = tent_shot_utils.PointNet_forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)
                        if (episodic_2):   
                            reset(args, config, logger)  
                    adapted_model.eval() 
                    if (original):
                        if (model_name == "Point_MAE"):
                            logits = adapted_model.module.classification_only(points_org, only_unmasked=False)     
                            logits_list.append(logits[None])   
                            if (episodic_2):     
                                reset(args, config, logger) 
                        elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                            logits = adapted_model.forward(points_org)    
                    else:                   
                        logits = adapted_model.module.classification_only(points[0], only_unmasked=False)     
                        
                logits = torch.cat(logits_list).mean(0)                  
                     
            ####################################### pred            
            target = labels.view(-1)
            if (method == "Tent" or method == "Tent_WA" or method == "Tent_Modified" or method == "Tent_WA2" or method == "Tent_WA_FPS" or method == "Tent_WA_Aug" 
                or method == "Tent_WA_FPS_Downsample" or method == "Ensemble" or method == "Tent_WA_Aug_FPS"):     
                pred = logits.argmax(-1).view(-1)
            elif (method == "Tent_Dino"): 
                logits = logits.reshape(target.shape[0], 6, -1)[:, 0, :]    
                pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach()) 
            if idx % 10 == 0:  
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)      

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.  
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',  
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)  
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:  
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',    
                  logger=logger)  
        acc_final += acc
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    acc_final = acc_final / len(corruptions)    
    print_log(f'\n\n########################## Mean Final Accuracy ::: {acc_final} ########\n\n', logger=logger)    
    f_write.close()
    if train_writer is not None:
        train_writer.close()


def tta_t3a(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    ext, cls = t3a_utils.get_cls_ext(base_model)  
    
    model_name = config.model.NAME  

    adapted_model = t3a_utils.T3A(args, ext, cls, config)

    args.severity = 5
    #f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_t3a_results/')
    f_write, logtime = get_writer_to_all_result(args, config, custom_path= args.experiment_path + "/")
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            logits = adapted_model(points)
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()


def tta_shot(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)
    #f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_shot_results/')
    f_write, logtime = get_writer_to_all_result(args, config, custom_path= args.experiment_path + "/")
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    
    model_name = config.model.NAME  
    
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            if (model_name == "Point_MAE"):     
                logits = tent_shot_utils.forward_and_adapt_shot(points, adapted_model, optimizer)
                
            elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                logits = tent_shot_utils.forward_and_adapt_shot_PointNet(points, adapted_model, optimizer)
                    
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()



def tta_LAME(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)
    #f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_shot_results/')
    f_write, logtime = get_writer_to_all_result(args, config, custom_path= args.experiment_path + "/")
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    
    model_name = config.model.NAME  
    
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            if (model_name == "Point_MAE"):     
                logits = tent_shot_utils.forward_and_adapt_shot(points, adapted_model, optimizer)
                
            elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                logits = tent_shot_utils.forward_and_adapt_shot_PointNet(points, adapted_model, optimizer)
                    
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()



def tta(args, config, train_writer=None):
    dataset_name = args.dataset_name
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
                f_write.write(f'Corruption LEVEL: {args.severity}' + '\n\n')

            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []
            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]
                args.grad_steps = 1

            method = config.model.transformer_config.method    

            for idx, (data, labels) in enumerate(tta_loader):

                if (method == "MATE" or method == "MATE_joint_MSE" or method == "MATE_joint_MSE_Downsample" or method == "MATE_our_joint_MSE_Downsample"):  
                    losses = AverageMeter(['Reconstruction Loss'])

                elif (method == "MATE_only_cls_KLD"):    
                    losses = AverageMeter(['KLD Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        points_list = []
                        points_list_org_low = []
                        if (method == "MATE_only_cls_KLD" or method == "MATE_joint_MSE_Downsample" or method == "MATE_our_joint_MSE_Downsample"):   
                            for i in range(6):
                                if (i < 6):
                                    points = pytorch3d.ops.sample_farthest_points(points, K=npoints, random_start_point= True)[0] 
                                    assert points.size(1) == npoints

                                    points_list.append(points[None])

                                else:
                                    points = pytorch3d.ops.sample_farthest_points(points, K=config.model.transformer_config.N_downsample, random_start_point= True)[0]
                                    points = torch.nn.functional.interpolate(points.permute(0, 2, 1)[..., None], size= (1024, 1), mode='bilinear') 
      
                                    points = points[..., 0].permute(0, 2, 1)     

                                    points_list.append(points[None])  

                            points = torch.cat((points_list), 0)      


                        ############################### Network
                        if (method == "MATE"):
                            loss_recon, loss_p_consistency, loss_regularize = base_model(points)
                            loss = loss_recon[0] + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency) 

                        elif (method == "MATE_joint_MSE" or method == "MATE_joint_MSE_Downsample" or method == "MATE_our_joint_MSE_Downsample"):
                            loss_recon, loss_p_consistency, mse_loss = base_model(points)
                            loss_recon = loss_recon[0] 
                            mse_loss = 1000. * mse_loss    
                            loss = loss_recon      # + (0.0001 * loss_p_consistency)        

                        elif (method == "MATE_only_cls_KLD"):
                            ret, kld_loss = base_model.module.classification_only_train(points, args.only_unmasked)         
                            kld_loss = kld_loss *1000.   
                            loss = kld_loss

                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        base_model.zero_grad()
                        optimizer.zero_grad()

                    else:
                        continue

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    if (method == "MATE"):
                        print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                f'GradStep - {grad_step} / {args.grad_steps},'
                                f'Reconstruction Loss {[l for l in losses.val()]}',
                                logger=logger)
                        
                    elif (method == "MATE_joint_MSE" or method == "MATE_joint_MSE_Downsample" or method == "MATE_our_joint_MSE_Downsample"):
                        print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                f'GradStep - {grad_step} / {args.grad_steps},'
                                f'Reconstruction_plus_MSE Loss {[l for l in losses.val()]}',    
                                logger=logger)    
                        
                    elif (method == "MATE_only_cls_KLD"):
                        print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                f'GradStep - {grad_step} / {args.grad_steps},'
                                f'KLD Loss {[l for l in losses.val()]}',
                                logger=logger)    

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model.module.classification_only(points, only_unmasked=False) 
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

                    acc_avg.append(acc.cpu())
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)  

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',  
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def tta_dua(args, config, train_writer=None):
    dataset_name = args.dataset_name
    # assert args.tta
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.disable_bn_adaptation = False

    # args.batch_size_tta = 48
    args.batch_size_tta = 1
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    model_name = config.model.NAME  

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                #f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')   
                f_write, logtime = get_writer_to_all_result(args, config, custom_path= args.experiment_path + "/")  
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            test_pred = []
            test_label = []
            base_model = load_base_model(args, config, logger)

            for idx, (data, labels) in enumerate(tta_loader):
                base_model.train()

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject', 'scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        if (model_name == "Point_MAE"): 
                            _ = base_model.module.classification_only(points,
                                                                    only_unmasked=True)  # only a forward pass through the encoder with BN in train mode
                        elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                            _ = base_model.module.forward(points)
                        # loss=0
                    else:
                        continue

                    # print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},',
                    #           logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                if (model_name == "Point_MAE"): 
                    logits = base_model.module.classification_only(points, only_unmasked=False)
                elif(model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):
                    logits = base_model.module.forward(points)   
                         
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 100 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def tta_partseg(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_partnet):
            root = config.root_partseg

            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if args.corruption != 'clean':
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}_{args.severity}')
            else:
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_part_seg/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_partnet}' + '\n\n')

            TEST_DATASET = tta_datasets.PartNormalDatasetSeg(root=root, npoints=config.npoint, split='test',
                                                             normal_channel=config.normal, debug=args.debug)
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_partnet) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()


def tta_shapenet(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_h5):
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_shape_net/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_h5}' + '\n\n')

            TEST_DATASET = tta_datasets.ShapeNetC(args,
                                                  root='./data/shapenet_c')
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_h5) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()

########################################################################################################################################################## Visualization
def loss_visualization_infer(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject' or dataset_name == 'scanobject_cvpr':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError
    
    layer_norm = config.model.transformer_config.layer_norm
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    npoints_ours = config.npoints_ours
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger) 
    if (layer_norm == False):
        adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)                                               
    else:
        adapted_model, optimizer = tent_shot_utils.setup_tent_shot_both(args, model=base_model)    
    #adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model, config= config)                               ### Added by Ali
    
    if dataset_name == 'modelnet' or dataset_name == 'scanobject' or dataset_name == 'scanobject_cvpr':     
        args.severity = 5
    if dataset_name == 'shapenetcore':       
        args.severity = 5 
        
    f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_tent_results/')
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    
    method = config.model.transformer_config.method
    episodic = config.model.transformer_config.reset   
    episodic_2 = config.model.transformer_config.reset_2
    n_aug = config.model.transformer_config.N_Aug   
    n_aug_2 = config.model.transformer_config.N_Aug_2   
    type_aug = config.model.transformer_config.Type_Aug  
    iteration = config.model.transformer_config.iteration   
    original = config.model.transformer_config.original   
    N_downsample = config.model.transformer_config.N_downsample
    Number_downsample = config.model.transformer_config.Number_downsample
    model_name = config.model.NAME
    cross_entropy = config.model.transformer_config.cross_entropy
     
    acc_final = 0
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            print("###########################################################################################", idx) 
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            
            ####################################### Data    
                
            points_list = []  
            #points_org = misc.fps(points, npoints)   
                
            if (model_name == "Point_MAE"):              
                for i in range(n_aug):  
                    points_org = points    
                    points = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0] 
                    points_list.append(points[None])  
                    
                points = torch.cat((points_list), 0)             
                
            if (model_name == "PointNet" or model_name == "DGCNN_cls" or model_name == "PointNet_ssg_Plus_Plus" or model_name == "CurveNet"):    
                for i in range(n_aug):  
                    points_fps = pytorch3d.ops.sample_farthest_points(points, K=npoints_ours, random_start_point= True)[0] 
                    points_list.append(points_fps[None])  
                        
                #if (dataset_name == "shapenetcore" or dataset_name == "scanobject"): 
                points_org =  points_fps
                points = torch.cat((points_list), 0)    
            
            ####################################### Network   
                
            if episodic:
                reset(args, config, logger)   
            adapted_model.train() 
            all_weights = []      
            for i in range(n_aug):         
            #for i in range(n_aug + Number_downsample):       
                logits, all_weights_cmp = tent_shot_utils.forward_and_adapt_tent_with_WA(points[i], adapted_model, optimizer, all_weights, iteration, layer_norm)
                if (i == 0):            
                    w1 = params_to_vector(adapted_model.parameters()).cpu()
                    
                if (i == 1):            
                    w2 = params_to_vector(adapted_model.parameters()).cpu()    
                    
                if (i == 2):            
                    w3 = params_to_vector(adapted_model.parameters()).cpu()
                    break
            if (corr_id == 0 and idx == 0):        
                G=None
                margin=0.6

                if G is None:
                    G = int((1.0 + margin*2) * 15)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs, labels = points[2].to(device, non_blocking=True), labels.to(device, non_blocking=True)   
                
                results = infer_grid_batch(w1, w2, w3, adapted_model, inputs, labels, G=G, margin=margin)


                save_dir = "/export/livia/home/vision/Abahri/projects/MATE_Org/MATE/model_weights/gaussian_noise_b12_vis_results_New_TENT.pth"
                torch.save(results, save_dir)   
                
                break     
        break
    
    print("done!")    




def evaluate_models(teloader=None, base_model=None, loaded_weights=None, teset=None, device=None, template=None):
   
    ### evaluate the perofmance of the WA model on the whole test set
    accs = []
    losses = []
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader), total=len(teloader)): 
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        b_acc, b_loss, _ = final_performance(inputs, labels, base_model, teset.classes, device, template)
        accs.append(b_acc)
        losses.append(b_loss)

    print(f"WA model performance on the whole test set: {np.mean(accs):0.4f}, {np.mean(losses):0.4f}")


    ### evaluate the perofmance of the first model on the whole test set
    accs = []
    losses = []
    base_model.load_state_dict(loaded_weights[0])
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader), total=len(teloader)): 
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        b_acc, b_loss, _ = final_performance(inputs, labels, base_model, teset.classes, device, template)
        accs.append(b_acc)
        losses.append(b_loss)

    print(f"Model 0 performance on the whole test set: {np.mean(accs):0.4f}, {np.mean(losses):0.4f}")

    ### evaluate the perofmance of the second model on the whole test set
    accs = []
    losses = []
    base_model.load_state_dict(loaded_weights[1])
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader), total=len(teloader)): 
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        b_acc, b_loss, _ = final_performance(inputs, labels, base_model, teset.classes, device, template)
        accs.append(b_acc)
        losses.append(b_loss)

    print(f"Model 1 performance on the whole test set: {np.mean(accs):0.4f}, {np.mean(losses):0.4f}")


    ### evaluate the perofmance of the second model on the whole test set
    accs = []
    losses = []
    base_model.load_state_dict(loaded_weights[2])
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader), total=len(teloader)): 
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        b_acc, b_loss, _ = final_performance(inputs, labels, base_model, teset.classes, device, template)
        accs.append(b_acc)
        losses.append(b_loss)

    print(f"Model 2 performance on the whole test set: {np.mean(accs):0.4f}, {np.mean(losses):0.4f}")


def set_global_seeds(seed_value=42):
    """Set global seeds to make experiments reproducible."""
    random.seed(seed_value)    # Python's builtin random module
    np.random.seed(seed_value) # Numpy library
    torch.manual_seed(seed_value) # Torch

    # if using torch's CUDA (GPU) capabilities:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def text_emb_ensemble(model, classnames, templates, K=None):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if K == None:
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings /= class_embeddings.norm()
            weights.append(class_embeddings)
        weights = torch.stack(weights, dim=1).cuda()
    return weights


def final_performance(inputs, labels, model, device):

    
    model.eval()
    model.cuda()
    logits = model.module.classification_only(inputs, only_unmasked=False)  
    
    target = labels.view(-1)
    pred = logits.argmax(-1).view(-1)
    
    acc = (target == pred).sum() / pred.shape[0] 

    ## calulate the CE loss between the raw similarity and the true labels
    
    with torch.no_grad():
        #loss = nn.CrossEntropyLoss()
        target = target.to(device)
        #loss_val = loss(logits, target)
        loss_val = softmax_entropy(logits).mean(0)      
        loss_val = loss_val.item()

    return acc, loss_val         

def params_to_vector(parameters):
    return torch.cat(list(map(lambda x: x.detach().flatten(), parameters)))



def argparser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', type=str, default='work/vis/t3_wa10iter')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='/export/livia/home/vision/Mnoori/data/')

    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--clip', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for base training')
    parser.add_argument('--level', default=5, type=int)
    parser.add_argument('--distributed', action='store_true', help='Activate distributed training')

    parser.add_argument('--full_eval', type=bool, default=False)

    return parser.parse_args()


############# visualization
def get_basis(w1, w2, w3):
    """https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py#L105

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
    """
    u = w2 - w1
    du = u.norm()
    u /= du

    v = w3 - w1
    v -= u.dot(v) * u
    dv = v.norm()
    v /= dv

    return u, v, du, dv


def get_xy(point, origin, vector_x, vector_y):
    return torch.as_tensor(
        [
            torch.dot(point - origin, vector_x),
            torch.dot(point - origin, vector_y)
        ]
    )

def infer_grid_batch(w1, w2, w3, base_model, inputs, labels, G, margin=0.2):
    """Make a grid by (w1, w2, w3)-plane and infer for each grid point.
    https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
        base_model: model for architecture.
        test_loader: dataloader for test env.
        G: n_grid_points (per axis); total points = G * G.
        margin
    """
    u, v, du, dv = get_basis(w1, w2, w3)

    alphas = np.linspace(0. - margin, 1. + margin, G)
    betas = np.linspace(0. - margin, 1. + margin, G)

    results = []

    for i, w in enumerate([w1, w2, w3]):
        c = get_xy(w, w1, u, v)

        results.append({
            "ij": f"w{i+1}",
            "grid_xy": c
        })

    tk = tqdm(total=G*G)
    device = "cuda"
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            tk.set_description(f"i={i+1}/{G}, j={j+1}/{G}")
            interpolated = w1 + alpha * du * u + beta * dv * v
            copy_flat_params_(interpolated, base_model)
            # update_bn -> skip

            acc, loss= final_performance(inputs, labels, base_model, device)
            #  c = get_xy(interpolated, w1, u, v)
            #  c == [alpha * dx, beta * dy] -> it has a little residual < 0.01.

            results.append({
                "ij": [i, j],
                "grid_xy": torch.as_tensor([alpha * du, beta * dv]),
                "error": 1. - acc,
                "loss": loss
            })
            tk.update(1)

    return results


def copy_flat_params_(flat_params, model):
    offset = 0
    for name, p in model.named_parameters():  # Use named_parameters to get names
        size = p.numel()
        ip = flat_params[offset:offset+size].view(p.shape)

        # Check if p is equal to ip and print the name if they are not
        # if not torch.allclose(p.cpu().float(), ip.cpu().float()):
        #     print(f"{name} is not equal to ip")  # Include the parameter name in the output

        with torch.no_grad():
            p.copy_(ip.to(p.dtype))  # Ensure ip is converted to p's dtype before copying
        offset += size

def softmax_entropy(x: torch.Tensor):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
##########################################################################################################################################################


##################
"""# Load a PLY file
ply_path = './outputs_ply/output_1.ply'
point_cloud = o3d.io.read_point_cloud(ply_path)   

# Convert Open3D.PointCloud to numpy array
points = np.asarray(point_cloud.points)

# Convert numpy array to PyTorch tensor
points_tensor = torch.tensor(points, dtype=torch.float32)

points = pytorch3d.ops.sample_farthest_points(points_tensor[None], K=256, random_start_point= True)[0] 

from plyfile import PlyData, PlyElement

data = points[0]

# Suppose 'data' is your 512x3 numpy array/tensor
# You can convert your tensor to a numpy array if it's not already, e.g.,
# data = tensor.numpy() if it's a PyTorch tensor or data = tensor.numpy() if it's a TensorFlow tensor

# Create a structured array to store the data for PLY format
vertices = np.array([(data[i, 0], data[i, 1], data[i, 2]) for i in range(data.shape[0])], 
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# Create a PlyElement instance to describe the vertex data
vertex_element = PlyElement.describe(vertices, 'vertex')

# Create PlyData instance and write to file
PlyData([vertex_element]).write('output_fps_5.ply')"""

