# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:05:58 2021

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import math

import glob
import os
import sys
import argparse
from time import time
from datetime import datetime
import random
import copy
import json
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from skimage import io, color, segmentation

from networks import JigsawUNetDown
from dataset import IXIdataset_Jigsaw

# import tensorflow # needs to call tensorflow before torch, otherwise crush
# sys.path.append('Utils')
# from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

# sys.path.append('Dataset')

from utils import adjust_learning_rate, compute_accuracy

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on MRI IXI dataset')
parser.add_argument('--datapath', type=str, help='Path to Imagenet folder')
parser.add_argument('--mri_view', default='sagittal', type=str, help='MRI view to use for training: [sagittal], [coronal], [axial]')
parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

# class MyDataParallel(nn.DataParallel):
#     def __getattr__(self, name):
#         return getattr(self.module, name)

def __retrive_permutations(classes):
    all_perm = np.load('permutations_mean_%d.npy'%(classes))
    # from range [1,9] to [0,8]
    if all_perm.min()==1:
        all_perm = all_perm-1

    return all_perm

def main():
    import sys
    sys.setrecursionlimit(2000)
    if args.gpu is not None:
        print(('Using GPU %s'%args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    else:
        print('CPU mode')
    
    print('Process number: %d'%(os.getpid()))

    permutations = __retrive_permutations(args.classes)

    timestamp_current = datetime.now()
    timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir='/home/qasymjomart/runs/runs_'+args.savename+'_'+timestamp_current)

    image_transformer = transforms.Compose([
                        # transforms.ToTensor(),
                        transforms.CenterCrop(225)
                        # transforms.RandomAffine(90)
                        ])

    augment_tile = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.Resize((75,75))
                ])
    
    # path is '/home/guest/qasymjomart/seg_mri/IXI-T1/IXI-T1-final/'
    train_dataset = IXIdataset_Jigsaw(root=args.datapath, 
                                view = args.mri_view, # !Important
                                mode='train',
                                permutations = permutations,
                                image_transforms_ = image_transformer,
                                tile_transforms_ = augment_tile)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                            shuffle=True, num_workers=0)
    
    iter_per_epoch = train_dataset.__len__()/args.batch


    print('No. of MRI Images (train dataloader size): train %d'%(train_dataset.__len__()))
    
    # Network initialize
    net = JigsawUNetDown(1, args.classes)
    if args.gpu is not None:
        net.cuda()
        # net = nn.DataParallel(net)
    
    ############## Load from checkpoint if exists, otherwise from model ###############
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f and args.mri_view in f and args.savename in f]
        if len(files)>0:
            files.sort()
            #print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ',ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)

    criterion = nn.CrossEntropyLoss()
    # !Important: consider changing to Adam
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)
    
    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,args.classes)))
    print(('Checkpoint: '+args.checkpoint))
    
    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        
        end = time()
        for i, (images, labels) in enumerate(train_dataloader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]
            
            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            writer.add_scalar(args.savename + ' ' + 'Pre-training Jigsaw loss', loss, steps)

            if steps%20==0:
                print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, args.epochs, steps, 
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss, acc)))

            steps += 1

            if steps%1000==0:
                filename = '%s%s_%s_%03i_%06d.pth.tar'%(args.checkpoint, args.savename, args.mri_view, epoch, steps)
                net.save(filename)
                print('Saved: '+args.checkpoint)
            
            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

if __name__ == "__main__":
    main()
