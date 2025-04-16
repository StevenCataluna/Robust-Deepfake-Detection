#-*- coding: utf-8 -*-
import os
import time
from numpy import arange

import torch
from torch.amp import autocast, GradScaler
import cv2
import numpy as np
from tqdm import tqdm
import random

from lib.metrics import get_acc_mesure_func, bin_calculate_auc_ap_ar
from logs.logger import board_writing
from package_utils.utils import debugging_panel


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count if self.count != 0 else 0
        
        
def compress_image(tensor_img, quality_range=(30,80)):
    # convert tensor to np array
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)      
    # get quality value in range
    quality = random.randint(quality_range[0], quality_range[1])
    # encode to jpg format and reverse
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # Convert to tensor
    compressed_img = torch.tensor(compressed_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return compressed_img.cuda()




def get_batch_data(batch_data):
    inputs = batch_data['img']
    labels = batch_data['label']
    targets = batch_data['target']
    heatmaps = batch_data['heatmap']
    cstency_heatmaps = None
    offsets = None
    
    if 'cstency_heatmap' in batch_data:
        cstency_heatmaps = batch_data['cstency_heatmap']

    if 'offset' in batch_data:
        offsets = batch_data['offset']
    
    return inputs, labels, targets, heatmaps, cstency_heatmaps, offsets

#ORIGINAL
def train_original(cfg, model, critetion, optimizer, epoch, data_loader, logger, writer, devices, trainIters, g_scaler, metrics_base='combine'):
    calculate_acc = get_acc_mesure_func(metrics_base)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    
    scaler = g_scaler

    #Switch to train mode
    model.float()
    model.train()
    data_loader = tqdm(data_loader, dynamic_ncols=True)
    start = time.time()
    for i, batch_data in enumerate(data_loader):
        inputs, labels, targets, heatmaps, cstency_heatmaps, offsets = get_batch_data(batch_data)
        inputs = inputs.cuda().to(non_blocking=True, dtype=torch.float32)
        #Measuring data loading time
        data_time.update(time.time() - start)
        
        loop = arange(1) if cfg.TRAIN.optimizer != 'SAM' else arange(2)
        for idx in loop:

            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs, _ = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[0]
            #In case outputs contain a dict key
            if isinstance(outputs, dict):
                outputs_hm = outputs['hm']
                outputs_cls = outputs['cls']
                outputs_offset = outputs['offset'] if 'offset' in outputs.keys() else None
                outputs_cstency = outputs['cstency'] if 'cstency' in outputs.keys() else None
                
                if idx == 0:
                    first_outputs_hm = outputs_hm
                    first_outputs_cls = outputs_cls
            
            if 'Combined' in cfg.TRAIN.loss.type:
                labels = labels.cuda().to(non_blocking=True, dtype=torch.float32)
                # labels = labels.cuda().to(non_blocking=True).long()
                
                if offsets is not None:
                    offsets = offsets.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cstency_heatmaps is not None:
                    cstency_heatmaps = cstency_heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cfg.TRAIN.loss.type != 'CombinedHeatmapBinaryLoss':
                    heatmaps = heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                else:
                    heatmaps = targets.cuda().to(non_blocking=True, dtype=torch.float32)
                    
                loss_ = critetion(outputs_hm, heatmaps, outputs_cls.sigmoid(), labels, 
                                  offset_preds=outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=outputs_cstency,
                                  cstency_gts=cstency_heatmaps)
                loss = loss_['hm']
                if 'cls' in loss_.keys():
                    loss += loss_['cls']
                if 'dst_hm_cls' in loss_.keys():
                    loss += loss_['dst_hm_cls']
                if 'offset' in loss_.keys():
                    loss += loss_['offset']
                if 'cstency' in loss_.keys():
                    loss += loss_['cstency']
            else:
                loss = critetion(outputs, heatmaps)
            

            scaler.scale(loss).backward()
            
            if cfg.TRAIN.optimizer != 'SAM':
                # optimizer.step()
                scaler.step(optimizer)
            else:
                if idx == 0:
                    scaler.step(optimizer.first_step(zero_grad=True))
                else:
                    scaler.step(optimizer.second_step(zero_grad=True))
            scaler.update()


                    
        if cfg.TRAIN.debug.active:
            debugging_panel(cfg.TRAIN.debug, inputs, heatmaps, first_outputs_hm, i, batch_cls_pred=first_outputs_cls)
        
        if metrics_base == 'binary':
            acc_ = calculate_acc(first_outputs_cls, targets=targets, labels=labels)
        elif metrics_base == 'heatmap':
            acc_ = calculate_acc(first_outputs_hm, targets=targets, labels=labels)
        else:
            acc_ = calculate_acc(first_outputs_hm, first_outputs_cls, targets=targets, labels=labels, cls_lamda=critetion.cls_lmda)
        
        if isinstance(inputs, list):
            batch_size = inputs[0].size(0)
        else:
            batch_size = inputs.size(0)
        
        #Measure accuracy and record loss
        losses.update(loss.item(), n=batch_size)
        acc.update(acc_, n=batch_size)
        
        batch_time.update(time.time() - start)
        start = time.time()
        
        #Logging
        if i % 5 == 0:
            params = {}
            if 'Combined' in cfg.TRAIN.loss.type:
                if hasattr(critetion, 'dst_hm_cls_lmda') and critetion.dst_hm_cls_lmda > 0:
                    params['loss_dst']=loss_['dst_hm_cls'].item()
                if hasattr(critetion, 'offset_lmda') and critetion.offset_lmda > 0:
                    params['loss_offset']=loss_['offset'].item()
                if hasattr(critetion, 'cstency_lmda') and critetion.cstency_lmda > 0:
                    params['loss_cstency']=loss_['cstency'].item()
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val, 
                                  loss_cls=loss_['cls'].item(), **params)
            else:
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val)
        
        trainIters += 1
        if cfg.TRAIN.tensorboard:
            board_writing(writer, losses.avg, acc.avg, trainIters, 'Train')
    return losses, acc, trainIters



#Mixed Precision 2 Inputs
def train(cfg, model, critetion, optimizer, epoch, data_loader, logger, writer, devices, trainIters, g_scaler, metrics_base='combine'):
    calculate_acc = get_acc_mesure_func(metrics_base)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    scaler = g_scaler
    #Switch to train mode
    model = model.float()
    model.train()
    data_loader = tqdm(data_loader, dynamic_ncols=True)
    start = time.time()
    # print(f"Optimizer: {optimizer}")  #Check if optimizer is None
    if optimizer is None:
        raise ValueError("Optimizer is None. Please provide a valid optimizer.")

    print("ITERATION STARTED")

    for i, batch_data in enumerate(data_loader):
        print("ITERATION " + str(i))
        
        inputs, labels, targets, heatmaps, cstency_heatmaps, offsets = get_batch_data(batch_data)
        inputs = inputs.cuda().to(non_blocking=True, dtype=torch.float32)

        #Create low quality inputs
        high_quality_inputs = inputs
        low_quality_inputs = torch.stack([compress_image(img) for img in high_quality_inputs])
        low_quality_inputs = low_quality_inputs.cuda().to(non_blocking=True, dtype=torch.float32)

        #Measuring data loading time
        data_time.update(time.time() - start)
        
        loop = arange(1) if cfg.TRAIN.optimizer != 'SAM' else arange(2)
        for idx in loop:
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                high_outputs, high_features = model(high_quality_inputs)
                low_outputs, low_features = model(low_quality_inputs)

            # print(f"Optimizer: {optimizer}")  # Check if optimizer is None
            if optimizer is None:
                raise ValueError("Optimizer is None. Please provide a valid optimizer.")

            if isinstance(high_outputs, list):
                high_outputs = high_outputs[0]
            
            if isinstance(low_outputs, list):
                low_outputs = low_outputs[0]


            if isinstance(high_outputs, dict):
                high_outputs_hm = high_outputs['hm']
                high_outputs_cls = high_outputs['cls']
                high_outputs_offset = high_outputs['offset'] if 'offset' in high_outputs.keys() else None
                high_outputs_cstency = high_outputs['cstency'] if 'cstency' in high_outputs.keys() else None

            if isinstance(high_outputs, dict):
                low_outputs_hm = low_outputs['hm']
                low_outputs_cls = low_outputs['cls']
                low_outputs_offset = low_outputs['offset'] if 'offset' in low_outputs.keys() else None
                low_outputs_cstency = low_outputs['cstency'] if 'cstency' in low_outputs.keys() else None

            
                if idx == 0:
                    first_outputs_hm = high_outputs_hm
                    first_outputs_cls = high_outputs_cls

            if 'Combined' in cfg.TRAIN.loss.type:
                labels = labels.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if offsets is not None:
                    offsets = offsets.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cstency_heatmaps is not None:
                    cstency_heatmaps = cstency_heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cfg.TRAIN.loss.type != 'CombinedHeatmapBinaryLoss':
                    heatmaps = heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                else:
                    heatmaps = targets.cuda().to(non_blocking=True, dtype=torch.float32)
                
                #Only calculate HSIC once
                high_loss_ = critetion(high_outputs_hm, heatmaps, high_outputs_cls.sigmoid(), labels, 
                                  offset_preds=high_outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=high_outputs_cstency,
                                  cstency_gts=cstency_heatmaps,
                                  features_high=high_features,
                                  features_low=low_features)

                high_loss = high_loss_['hm']
                if 'cls' in high_loss_.keys():
                    high_loss += high_loss_['cls']
                if 'dst_hm_cls' in high_loss_.keys():
                    high_loss += high_loss_['dst_hm_cls']
                if 'offset' in high_loss_.keys():
                    high_loss += high_loss_['offset']
                if 'cstency' in high_loss_.keys():
                    high_loss += high_loss_['cstency']
                # if 'hsic' in high_loss_.keys():
                #     high_loss += high_loss_['hsic']
                # else:
                #     high_loss = critetion(high_outputs, heatmaps)

                low_loss_ = critetion(low_outputs_hm, heatmaps, low_outputs_cls.sigmoid(), labels, 
                                  offset_preds=low_outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=low_outputs_cstency,
                                  cstency_gts=cstency_heatmaps,
                                  features_high=None,
                                  features_low=None)
            
                low_loss = low_loss_['hm']
                if 'cls' in low_loss_.keys():
                    low_loss += low_loss_['cls']
                if 'dst_hm_cls' in low_loss_.keys():
                    low_loss += low_loss_['dst_hm_cls']
                if 'offset' in low_loss_.keys():
                    low_loss += low_loss_['offset']
                if 'cstency' in low_loss_.keys():
                    low_loss += low_loss_['cstency']
                # if 'hsic' in low_loss_.keys():
                #     low_loss += low_loss_['hsic']
                
                #Adding all losses together
                loss = (high_loss + low_loss) / 2 + high_loss_['hsic']

            else:
                low_loss = critetion(low_outputs, heatmaps)
                high_loss = critetion(high_outputs, heatmaps)
                loss = low_loss + high_loss

            # print(f"Optimizer: {optimizer}")  # Check if optimizer is None
            if optimizer is None:
                raise ValueError("Optimizer is None. Please provide a valid optimizer.")

            # optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            
            if cfg.TRAIN.optimizer != 'SAM':
                # optimizer.step()
                scaler.step(optimizer)
            else:
                if idx == 0:
                    scaler.step(optimizer.first_step(zero_grad=True))
                else:
                    scaler.step(optimizer.second_step(zero_grad=True))
            scaler.update()
                    
        if cfg.TRAIN.debug.active:
            debugging_panel(cfg.TRAIN.debug, inputs, heatmaps, first_outputs_hm, i, batch_cls_pred=first_outputs_cls)
        
        if metrics_base == 'binary':
            acc_ = calculate_acc(first_outputs_cls, targets=targets, labels=labels)
        elif metrics_base == 'heatmap':
            acc_ = calculate_acc(first_outputs_hm, targets=targets, labels=labels)
        else:
            acc_ = calculate_acc(first_outputs_hm, first_outputs_cls, targets=targets, labels=labels, cls_lamda=critetion.cls_lmda)
        
        if isinstance(inputs, list):
            batch_size = inputs[0].size(0)
        else:
            batch_size = inputs.size(0)
        
        #Measure accuracy and record loss
        losses.update(loss.item(), n=batch_size)
        acc.update(acc_, n=batch_size)
        
        batch_time.update(time.time() - start)
        start = time.time()
        
        #Logging
        if i % 5 == 0:
            params = {}
            if 'Combined' in cfg.TRAIN.loss.type:
                if hasattr(critetion, 'dst_hm_cls_lmda') and critetion.dst_hm_cls_lmda > 0:
                    params['loss_dst']=high_loss_['dst_hm_cls'].item() + low_loss_['dst_hm_cls'].item()
                if hasattr(critetion, 'offset_lmda') and critetion.offset_lmda > 0:
                    params['loss_offset']=high_loss_['offset'].item() + low_loss_['offset'].item()
                if hasattr(critetion, 'cstency_lmda') and critetion.cstency_lmda > 0:
                    params['loss_cstency']=high_loss_['cstency'].item() + low_loss_['cstency'].item()
                if hasattr(critetion, 'hsic_lmda') and critetion.hsic_lmda > 0:
                    params['loss_hsic']=high_loss_['hsic'].item()
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val, 
                                  loss_cls=high_loss_['cls'].item(), **params)
            else:
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val)
        
        trainIters += 1
        if cfg.TRAIN.tensorboard:
            board_writing(writer, losses.avg, acc.avg, trainIters, 'Train')
        
        # if i == 5:
        #     return losses, acc, trainIters
    return losses, acc, trainIters




#Mixed Precision
def validate(cfg, model, critetion, epoch, data_loader, logger, writer, devices, valIters, metrics_base='combine'):
    calculate_acc = get_acc_mesure_func(metrics_base)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    scaler = GradScaler()
    model = model.float()

    #Switch to test mode
    model.eval()
    data_loader = tqdm(data_loader, dynamic_ncols=True)
    start = time.time()
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            inputs, labels, targets, heatmaps, cstency_heatmaps, offsets = get_batch_data(batch_data)
            inputs = inputs.cuda().to(non_blocking=True, dtype=torch.float32)
            #Create low quality inputs
            high_quality_inputs = inputs
            low_quality_inputs = torch.stack([compress_image(img) for img in high_quality_inputs])
            low_quality_inputs = low_quality_inputs.cuda().to(non_blocking=True, dtype=torch.float32)

            #Measuring data loading time
            data_time.update(time.time() - start)
            
            with autocast(device_type='cuda'):
                high_outputs, high_features = model(high_quality_inputs)
                low_outputs, low_features = model(low_quality_inputs)


            if isinstance(high_outputs, list):
                high_outputs = high_outputs[0]
            
            if isinstance(low_outputs, list):
                low_outputs = low_outputs[0]


            if isinstance(high_outputs, dict):
                high_outputs_hm = high_outputs['hm']
                high_outputs_cls = high_outputs['cls']
                high_outputs_offset = high_outputs['offset'] if 'offset' in high_outputs.keys() else None
                high_outputs_cstency = high_outputs['cstency'] if 'cstency' in high_outputs.keys() else None

            if isinstance(low_outputs, dict):
                low_outputs_hm = low_outputs['hm']
                low_outputs_cls = low_outputs['cls']
                low_outputs_offset = low_outputs['offset'] if 'offset' in low_outputs.keys() else None
                low_outputs_cstency = low_outputs['cstency'] if 'cstency' in low_outputs.keys() else None

            if 'Combined' in cfg.TRAIN.loss.type:
                labels = labels.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if offsets is not None:
                    offsets = offsets.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cstency_heatmaps is not None:
                    cstency_heatmaps = cstency_heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cfg.TRAIN.loss.type != 'CombinedHeatmapBinaryLoss':
                    heatmaps = heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                else:
                    heatmaps = targets.cuda().to(non_blocking=True, dtype=torch.float32)
                    

                # only calculate hsic once
                high_loss_ = critetion(high_outputs_hm, heatmaps, high_outputs_cls.sigmoid(), labels, 
                                  offset_preds=high_outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=high_outputs_cstency,
                                  cstency_gts=cstency_heatmaps,
                                  features_high = high_features,
                                  features_low = low_features)

                high_loss = high_loss_['hm']
                if 'cls' in high_loss_.keys():
                    high_loss += high_loss_['cls']
                if 'dst_hm_cls' in high_loss_.keys():
                    high_loss += high_loss_['dst_hm_cls']
                if 'offset' in high_loss_.keys():
                    high_loss += high_loss_['offset']
                if 'cstency' in high_loss_.keys():
                    high_loss += high_loss_['cstency']
                # if 'hsic' in high_loss_.keys():
                #     high_loss += high_loss_['hsic']
                # else:
                #     high_loss = critetion(high_outputs, heatmaps)

                low_loss_ = critetion(low_outputs_hm, heatmaps, low_outputs_cls.sigmoid(), labels, 
                                  offset_preds=low_outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=low_outputs_cstency,
                                  cstency_gts=cstency_heatmaps)
            
                low_loss = low_loss_['hm']
                if 'cls' in low_loss_.keys():
                    low_loss += low_loss_['cls']
                if 'dst_hm_cls' in low_loss_.keys():
                    low_loss += low_loss_['dst_hm_cls']
                if 'offset' in low_loss_.keys():
                    low_loss += low_loss_['offset']
                if 'cstency' in low_loss_.keys():
                    low_loss += low_loss_['cstency']
                # if 'hsic' in low_loss_.keys():
                #     low_loss += low_loss_['hsic']
                

                #Adding all losses together
                loss = (high_loss + low_loss) / 2 + high_loss_['hsic']

            else:
                low_loss = critetion(low_outputs, heatmaps)
                high_loss = critetion(high_outputs, heatmaps)
                loss = low_loss + high_loss 

            
            if cfg.TRAIN.debug.active:
                debugging_panel(cfg.TRAIN.debug, inputs, heatmaps, high_outputs_hm+low_outputs_hm, i, batch_cls_pred=high_outputs_cls+low_outputs_cls, split='val')
            
            if metrics_base == 'binary':
                acc_ = calculate_acc(high_outputs_cls+low_outputs_cls, targets=targets, labels=labels)
            elif metrics_base == 'heatmap':
                acc_ = calculate_acc(high_outputs_hm+low_outputs_hm, targets=targets, labels=labels)
            else:
                acc_ = calculate_acc(high_outputs_hm+low_outputs_hm, high_outputs_cls+low_outputs_cls, targets=targets, labels=labels, cls_lamda=critetion.cls_lmda)
            
            if isinstance(inputs, list):
                batch_size = inputs[0].size(0)
            else:
                batch_size = inputs.size(0)
            
            #Measure accuracy and record loss
            losses.update(loss.item(), n=batch_size)
            acc.update(acc_, n=batch_size)
            
            batch_time.update(time.time() - start)
            start = time.time()
            
            valIters += 1
            if cfg.TRAIN.tensorboard:
                board_writing(writer, losses.avg, acc.avg, valIters, 'Val')

            #Logging
            # params = {}
            # if 'Combined' in cfg.TRAIN.loss.type:
            #     if hasattr(critetion, 'dst_hm_cls_lmda') and critetion.dst_hm_cls_lmda > 0:
            #         params['loss_dst']=loss_['dst_hm_cls'].item()
            #     if hasattr(critetion, 'offset_lmda') and critetion.offset_lmda > 0:
            #         params['loss_offset']=loss_['offset'].item()
            #     if hasattr(critetion, 'cstency_lmda') and critetion.cstency_lmda > 0:
            #         params['loss_cstency']=loss_['cstency'].item()
            #     logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
            #                       data_time=data_time, losses=losses, acc=acc, 
            #                       speed=batch_size/batch_time.val, 
            #                       loss_cls=loss_['cls'].item(), **params)
            # else:
            #     logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
            #                       data_time=data_time, losses=losses, acc=acc, 
            #                       speed=batch_size/batch_time.val)
                

            params = {}
            if 'Combined' in cfg.TRAIN.loss.type:
                if hasattr(critetion, 'dst_hm_cls_lmda') and critetion.dst_hm_cls_lmda > 0:
                    params['loss_dst']=high_loss_['dst_hm_cls'].item() + low_loss_['dst_hm_cls'].item()
                if hasattr(critetion, 'offset_lmda') and critetion.offset_lmda > 0:
                    params['loss_offset']=high_loss_['offset'].item() + low_loss_['offset'].item() 
                if hasattr(critetion, 'cstency_lmda') and critetion.cstency_lmda > 0:
                    params['loss_cstency']=high_loss_['cstency'].item() + low_loss_['cstency'].item()
                if hasattr(critetion, 'hsic_lmda') and critetion.hsic_lmda > 0:
                    params['loss_hsic']=high_loss_['hsic'].item()
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val, 
                                  loss_cls=high_loss_['cls'].item(), **params)
            else:
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val)
            # if i == 5:
            #     return losses, acc, valIters
            break
    return losses, acc, valIters

# Original
def validate_original(cfg, model, critetion, epoch, data_loader, logger, writer, devices, valIters, metrics_base='combine'):
    calculate_acc = get_acc_mesure_func(metrics_base)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model = model.float()
    
    #Switch to test mode
    model.eval()
    data_loader = tqdm(data_loader, dynamic_ncols=True)
    start = time.time()
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            inputs, labels, targets, heatmaps, cstency_heatmaps, offsets = get_batch_data(batch_data)
            inputs = inputs.to(devices, non_blocking=True, dtype=torch.float32).cuda()
            #Measuring data loading time
            data_time.update(time.time() - start)
            with autocast(device_type='cuda'):
                outputs, _ = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            #In case outputs contain a dict key
            if isinstance(outputs, dict):
                outputs_hm = outputs['hm']
                outputs_cls = outputs['cls']
                outputs_offset = outputs['offset'] if 'offset' in outputs.keys() else None
                outputs_cstency = outputs['cstency'] if 'cstency' in outputs.keys() else None

            if 'Combined' in cfg.TRAIN.loss.type:
                labels = labels.cuda().to(non_blocking=True, dtype=torch.float32)
                # labels = labels.cuda().to(non_blocking=True).long()
                
                if offsets is not None:
                    offsets = offsets.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cstency_heatmaps is not None:
                    cstency_heatmaps = cstency_heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                
                if cfg.TRAIN.loss.type != 'CombinedHeatmapBinaryLoss':
                    heatmaps = heatmaps.cuda().to(non_blocking=True, dtype=torch.float32)
                else:
                    heatmaps = targets.cuda().to(non_blocking=True, dtype=torch.float32)
                    
                loss_ = critetion(outputs_hm, heatmaps, outputs_cls.sigmoid(), labels, 
                                  offset_preds=outputs_offset, 
                                  offset_gts=offsets,
                                  cstency_preds=outputs_cstency,
                                  cstency_gts=cstency_heatmaps)
                loss = loss_['hm']
                if 'cls' in loss_.keys():
                    loss += loss_['cls']
                if 'dst_hm_cls' in loss_.keys():
                    loss += loss_['dst_hm_cls']
                if 'offset' in loss_.keys():
                    loss += loss_['offset']
                if 'cstency' in loss_.keys():
                    loss += loss_['cstency']
            else:
                loss = critetion(outputs, heatmaps)
            
            if cfg.TRAIN.debug.active:
                debugging_panel(cfg.TRAIN.debug, inputs, heatmaps, outputs_hm, i, batch_cls_pred=outputs_cls, split='val')
            
            if metrics_base == 'binary':
                acc_ = calculate_acc(outputs_cls, targets=targets, labels=labels)
            elif metrics_base == 'heatmap':
                acc_ = calculate_acc(outputs_hm, targets=targets, labels=labels)
            else:
                acc_ = calculate_acc(outputs_hm, outputs_cls, targets=targets, labels=labels, cls_lamda=critetion.cls_lmda)
            
            if isinstance(inputs, list):
                batch_size = inputs[0].size(0)
            else:
                batch_size = inputs.size(0)
            
            #Measure accuracy and record loss
            losses.update(loss.item(), n=batch_size)
            acc.update(acc_, n=batch_size)
            
            batch_time.update(time.time() - start)
            start = time.time()
            
            valIters += 1
            if cfg.TRAIN.tensorboard:
                board_writing(writer, losses.avg, acc.avg, valIters, 'Val')

            #Logging
            params = {}
            if 'Combined' in cfg.TRAIN.loss.type:
                if hasattr(critetion, 'dst_hm_cls_lmda') and critetion.dst_hm_cls_lmda > 0:
                    params['loss_dst']=loss_['dst_hm_cls'].item()
                if hasattr(critetion, 'offset_lmda') and critetion.offset_lmda > 0:
                    params['loss_offset']=loss_['offset'].item()
                if hasattr(critetion, 'cstency_lmda') and critetion.cstency_lmda > 0:
                    params['loss_cstency']=loss_['cstency'].item()
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val, 
                                  loss_cls=loss_['cls'].item(), **params)
            else:
                logger.epochInfor(epoch, i, len(data_loader), batch_time=batch_time, 
                                  data_time=data_time, losses=losses, acc=acc, 
                                  speed=batch_size/batch_time.val)
    return losses, acc, valIters


def test(cfg, model, critetion, epoch, data_loader, logger, writer, devices, valIters, metrics_base='combine'):
    calculate_acc = get_acc_mesure_func(metrics_base)
    total_preds = torch.tensor([]).cuda().to(dtype=torch.float64)
    total_labels = torch.tensor([]).cuda().to(dtype=torch.float64)
    
    #Switch to test mode
    model.eval()
    test_dataloader = tqdm(data_loader, dynamic_ncols=True)
    with torch.no_grad():
        for b, (inputs, labels, vid_ids) in enumerate(test_dataloader):
            inputs = inputs.to(dtype=torch.float64).cuda()
            labels = labels.to(dtype=torch.float64).cuda()
            
            outputs = model(inputs)
            # Applying Flip test
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            #In case outputs contain a dict key
            if isinstance(outputs, dict):
                hm_outputs = outputs['hm']
                cls_outputs = outputs['cls']

            total_preds = torch.cat((total_preds, cls_outputs), 0)
            total_labels = torch.cat((total_labels, labels), 0)
            
        acc_ = calculate_acc(total_preds, targets=None, labels=total_labels, threshold=cfg.TEST.threshold)
        auc_, ap_, ar_, mf1_ = bin_calculate_auc_ap_ar(total_preds, total_labels, metrics_base=metrics_base)
        
        logger.info(f'Current ACC, AUC, AP, AR, mF1 for {cfg.DATASET.DATA.TEST.FAKETYPE} --- {cfg.DATASET.DATA.TEST.LABEL_FOLDER} -- \
            {acc_*100} -- {auc_*100} -- {ap_*100} -- {ar_*100} -- {mf1_*100}')

    return acc_, auc_, ap_, ar_
