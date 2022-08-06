# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:18:50 2021

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
import math

import glob
import os
import sys
import json
import random
import argparse
import copy
from datetime import datetime
from time import time
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from skimage import io, color, segmentation
from sklearn.model_selection import KFold

from utils import adjust_learning_rate, compute_accuracy

from networks import UNet
from dataset import Decathlon
from loss import DiceLoss, DiceBCELoss

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train (finetune) UNet on MRI HarP dataset')
parser.add_argument('--datapath', type=str, help='Path to MRI images data folder')
parser.add_argument('--labelspath', type=str, help='Path to labels folder')
parser.add_argument('--metapath', type=str, help='Path to meta file')
parser.add_argument('--mri_view', default='sagittal', type=str, help='MRI view to use for training: [sagittal], [coronal], [axial]')
parser.add_argument('--pre_trained_model_path', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--gpu', default='0', type=str, help='gpu id(s)')
parser.add_argument('--seed', default=0, type=int, help='torch seed')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--checkpoint', default='finetune_checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--savename', default='', type=str, help='Save name, experiment name to save all files')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set, No training')
args = parser.parse_args()

def main():
	# GPU devices
	if args.gpu is not None:
		print(('Using GPU %s'%args.gpu))
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	else:
		print('CPU mode')
	
	if args.seed == 42:
		seed_numpy = np.random.randint(0, 10000)
	else:
		seed_numpy = args.seed
	print('SEED: ', seed_numpy)
	torch.manual_seed(seed_numpy)
	torch.cuda.manual_seed(seed_numpy)
	torch.cuda.manual_seed_all(seed_numpy) # if use multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	np.random.seed(seed_numpy)
	random.seed(seed_numpy)
	# torch.manual_seed(args.seed)

	image_transforms = transforms.Compose([
						transforms.CenterCrop(32)
						])
	
	# print('Process number: %d'%(os.getpid()))
	finetune_bl = True
	label_fractions = [1, 0.5, 0.2, 0.1, 0.05]

	timestamp_current = datetime.now()
	timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
	writer = SummaryWriter(log_dir='/home/qasymjomart/runs/runs_decathlon_finetune_' + args.savename + '_' + timestamp_current)

	gtlists_data = natsorted(glob.glob(args.datapath + '*.nii.gz'))
	test_vol_idxs = random.sample(gtlists_data, round(len(gtlists_data) * 0.10))
	train_vol_idxs = copy.deepcopy(gtlists_data)
	for vol in test_vol_idxs:
		train_vol_idxs.remove(vol)

	dataset_split_idxs = {"test": test_vol_idxs,
						"train": train_vol_idxs}
	dataset_split_idxs_filename = "dataset_split_idxs_" + args.savename + "_" + timestamp_current + ".json"

	# Saving the randomly selected idxs of the train and test set volumes as a json file
	with open(dataset_split_idxs_filename, "w") as fp:
		json.dump(dataset_split_idxs, fp, indent=4)

	with open(dataset_split_idxs_filename, "r") as fp:
		dataset_split_idxs_loaded = json.load(fp)

	for label_fraction in label_fractions:
		train_idxs = random.sample(dataset_split_idxs_loaded['train'], round(len(dataset_split_idxs_loaded['train']) * label_fraction))
		test_idxs = dataset_split_idxs_loaded['test']
	
		for finetune_type in ['linear', 'finetune', 'tfs']:
			print('Finetuning type: ', finetune_type)

			# Dataset and Dataloader
			train_dataset = Decathlon(
				dataroot=args.datapath,
				labelsroot=args.labelspath,
				metaroot=args.metapath,
				indexes=train_idxs,
				view=args.mri_view,
				mode='train',
				transforms_ = image_transforms
			)

			test_dataset = Decathlon(
				dataroot=args.datapath,
				labelsroot=args.labelspath,
				indexes=test_idxs,
				view=args.mri_view,
				mode='test',
				transforms_ = image_transforms
			)

			train_loader = DataLoader(
				train_dataset,
				batch_size=args.batch_size,
				shuffle=True,
				num_workers=0,
				pin_memory=True
			)

			test_loader = DataLoader(
				test_dataset,
				batch_size=1,
				shuffle=True,
				num_workers=0,
				pin_memory=True
			)
		
			# Model initialization and transferring of the pre-trained weights
			unet = UNet(n_channels=1, n_classes=1, bilinear=True)
			unet_state = unet.state_dict()

			# transferring the pre-trained SS weights to unet encoder part
			# decoder part is randomly initialized by xavier_normal_, see networks.py, apply_weights functions
			if finetune_type == 'finetune' or finetune_type == 'linear':        
				pre_trained_network = torch.load(args.pre_trained_model_path)
				for name, param in pre_trained_network.items():
					if name in unet_state:
						unet_state[name].copy_(param)
						if finetune_type == 'linear':
							try:
								unet.get_parameter(name).requires_grad = False
							except:
								continue

			# Loss function and optimizer
			dice_loss = DiceLoss() # dice loss
			dice_bce_loss = DiceBCELoss() # dice loss with binary-cross entropy
			optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
			
			if args.gpu is not None and torch.cuda.is_available():
				unet.cuda()
				dice_bce_loss.cuda()
				dice_loss.cuda()
				unet = nn.DataParallel(unet)
			
			# Checkpoints
			############## Load from checkpoint if exists, otherwise from model ###############
			if os.path.exists(args.checkpoint):
				files = [f for f in os.listdir(args.checkpoint) if args.savename in f and args.mri_view in f and 'pth' in f and finetune_type in f and str(label_fraction * 100) in f]
				if len(files)>0:
					files.sort()
					#print files
					ckp = files[-1]
					unet.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
					args.iter_start = int(ckp.split(".")[-3].split("_")[-2])
					print('Starting from: ',ckp)

			# Training
			print(('Start training: lr %f, batch size %d'%(args.lr,args.batch_size)))
			print(('Checkpoint: '+args.checkpoint))

			iter_per_epoch = train_dataset.__len__()/args.batch_size
			print('Iter per epoch: ', iter_per_epoch)
			
			# Train the Model
			batch_time, net_time = [], []
			steps = args.iter_start

			for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):

				if epoch%10==0 and epoch>0:
					logger_test = None
					# test(net,criterion,logger_test,val_loader,steps)
				lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
				
				end = time()
				for i, (images, masks) in enumerate(train_loader):
					batch_time.append(time()-end)
					if len(batch_time)>100:
						del batch_time[0]
					
					images = Variable(images)
					masks = Variable(masks)
					if args.gpu is not None:
						images = images.cuda()
						masks = masks.cuda()

					# Forward + Backward + Optimize
					optimizer.zero_grad(set_to_none=True)
					t = time()
					outputs = unet(images)

					net_time.append(time()-t)
					if len(net_time)>100:
						del net_time[0]
					
					# Dice coefficient (just used to measure the performance)
					train_dice = dice_loss(outputs, masks)

					# this measurement is used for training (backprop)
					loss = dice_bce_loss(outputs, masks)
					loss.backward()
					optimizer.step()
					loss = float(loss.cpu().data.numpy())

					writer.add_scalar(finetune_type + '_' + str(label_fraction * 100) + '_' + args.savename + ' Jigsaw Dice coeff ', 1-train_dice, steps)
					writer.add_scalar(finetune_type + '_' + str(label_fraction * 100) + '_' + args.savename + ' Jigsaw Dice loss ', loss, steps)    

					if steps%20==0:
						# printing status during training
						print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Dice Loss: %1.3f, Dice coefficient %1.3f' %(
									epoch+1, args.epochs, steps, 
									np.mean(batch_time), np.mean(net_time),
									lr, loss, 1-train_dice)))
						fold_training_file = open('logs/'+args.savename + '_' + finetune_type + '_' + str(label_fraction * 100) + '_' + '_training_log.txt', 'a')
						fold_training_file.write(str(loss) + '\n')
						fold_training_file.close()
						fold_training_performance_file = open('logs/'+args.savename + '_' + finetune_type + '_' + str(label_fraction * 100) + '_' + '_training_performance_log.txt', 'a')
						fold_training_performance_file.write(str(1-float(train_dice.cpu().data.numpy())) + '\n')
						fold_training_performance_file.close()
					
					steps += 1

					if steps%500==0:
						filename = '%s%s_%s_%s_%s_%03i_%06d.pth.tar'%(args.checkpoint, args.savename, args.mri_view, finetune_type, str(label_fraction * 100), epoch, steps)
						torch.save(unet.module.state_dict(), filename)
						# unet.save(filename)
						print('Saved: ' + args.checkpoint)
					
					end = time()

				if os.path.exists(args.checkpoint+'stop.txt'):
					# break without using CTRL+C
					break

				if os.path.exists(args.checkpoint+'pdb.txt'):
					import pdb; pdb.set_trace()
			
			# testing at the end of each k-fold training
			test(unet, test_loader, args.mri_view, label_fraction, finetune_type, args.savename)
			args.iter_start = 0

# Test
def test(net, test_loader, mri_view, label_fraction, finetune_type, savename):
	print('Evaluating network....... ')
	net.eval()
	dice_loss = DiceLoss()
	for i, (images, masks) in enumerate(test_loader):
		images = Variable(images)
		if args.gpu is not None:
			images = images.cuda()

		# Forward + Backward + Optimize
		outputs = net(images)
		outputs = outputs.cpu().data

		# Calculate dice coefficient
		test_loss = dice_loss(outputs, masks)

	print('Test Dice Coefficient %.3f%%' %(1-test_loss))
	f = open('tests/test_' + savename + '_' + '_' + str(label_fraction * 100) + '_'  + mri_view + '_' + finetune_type + '.txt', 'a')
	f.write(finetune_type + '_' + str(label_fraction * 100) + '_' + str((1 - test_loss).cpu().numpy()) + '\n')
	f.close()
	net.train()

if __name__ == "__main__":
	main()
