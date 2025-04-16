#-*- coding: utf-8 -*-
from __future__ import absolute_import
import time

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from configs.get_config import load_config
from models import *
from datasets import *
from losses import *
from logs.logger import Logger, LOG_DIR

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import os
import time
from numpy import arange

import torch
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import random
import argparse

import cv2

def parse_args(args=None):
    arg_parser = argparse.ArgumentParser('t-SNE feature analysis')
    arg_parser.add_argument('--cfg', '-c', help='Config file', required=True)
    args = arg_parser.parse_args(args)
    
    return args

def compress_image(tensor_img, quality=70, quality_range=(30,80)):
    """ Compress an image tensor using JPEG compression """
    # Convert tensor to numpy array (H, W, C)
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)  # Convert to 8-bit image
    
    # quality = random.randint(quality_range[0], quality_range[1])
    # Encode to JPEG format and decode back
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # Convert back to tensor and normalize
    compressed_img = torch.tensor(compressed_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return compressed_img.cuda()


def t_sne_analysis(features, labels, title="t-SNE Visualization of Test Features"):
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Ensure labels are a 1D array
    labels = np.array(labels).flatten()

    # Define colors and labels
    color_map = {0: 'blue', 1: 'red'}
    label_map = {0: "Real", 1: "Fake"}
    colors = np.array([color_map[int(label)] for label in labels])  # Convert labels to int

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7)

    # Create legend manually
    legend = [plt.Line2D([0], [0], marker='o', linestyle='None', color=color, markersize=10) 
               for key, color in color_map.items()]
    legend_labels = [label_map[key] for key in color_map.keys()]
    
    plt.legend(legend, legend_labels, title="Data Classes")
    
    plt.title('t-SNE analysis with HSIC Model')
    
    path = '/content/drive/MyDrive/LAA-Net/LAA-Net-Temp/Feature_plots/plot.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    if sys.argv[1:] is not None:
        args = sys.argv[1:]
    else:
        args = sys.argv[:-1]
    args = parse_args(args)
    
    # Loading config file
    cfg = load_config(args.cfg)
    logger = Logger(task='testing')

    #Seed
    seed = cfg.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    task = cfg.TEST.subtask
    flip_test = cfg.TEST.flip_test
    logger.info('Flip Test is used --- {}'.format(flip_test))


    if task == 'eval' and cfg.DATASET.DATA.TEST.FROM_FILE:
        assert cfg.DATASET.DATA.TEST.ANNO_FILE is not None, "Annotation file can not be None with evaluation test mode!"
        assert len(cfg.DATASET.DATA.TEST.ANNO_FILE), "Annotation file can not be empty with evaluation test mode!"
    
    device_count = torch.cuda.device_count()

    # build and load/initiate pretrained model
    model = build_model(cfg.MODEL, MODELS).to(torch.float32)
    logger.info('Loading weight ... {}'.format(cfg.TEST.pretrained))
    model = load_pretrained(model, cfg.TEST.pretrained)

    if device_count >= 1:
        model = nn.DataParallel(model, device_ids=cfg.TEST.gpus).cuda()
    else:
        model = model.cuda()

    
    start_loading = time.time()
    test_dataset = build_dataset(cfg.DATASET, 
                                     DATASETS,
                                     default_args=dict(split='test', config=cfg.DATASET))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                 shuffle=True,
                                 num_workers=cfg.DATASET.NUM_WORKERS)
    logger.info('Dataset loading time --- {}'.format(time.time() - start_loading))

    model.eval()

    test_dataloader = tqdm(test_dataloader, dynamic_ncols=True)
    features_list = []
    labels_list = []

    with torch.no_grad():
            for b, (inputs, labels, vid_ids) in enumerate(test_dataloader):
                i_st = time.time()
                if device_count > 0:
                    inputs = inputs.to(dtype=torch.float32).cuda()
                    labels = labels.to(dtype=torch.float32).cuda()

                inputs = torch.stack([compress_image(img, 30) for img in inputs])
                inputs = inputs.cuda().to(non_blocking=True, dtype=torch.float32)
                
                with autocast(device_type='cuda'):
                    outputs, features = model(inputs)
                
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                # if b == 2:
                #     break


    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    t_sne_analysis(features_all, labels_all)

