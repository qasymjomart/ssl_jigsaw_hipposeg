# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:18:50 2021

@author: qasymjomart
"""

from __future__ import print_function, division
import os
import glob
import numpy as np
import pandas as pd
from skimage import io, color, segmentation
import nibabel as nib
from natsort import natsorted

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

"""
This file load the MRI datasets and builds dataloaders on top of them
"""

class IXIdataset_Jigsaw(Dataset):
    def __init__(self, root='/home/guest/qasymjomart/seg_mri/IXI-T1/IXI-T1-final/', 
                       view = 'sagittal',
                       slices_to_use = (40, 110),
                       mode='train',
                       permutations = None,
                       image_transforms_ = None,
                       tile_transforms_ = None):
        """
        This __init__ function initializes the class.
        root: is the root to the nii files to be used in training or testing
        view: is the MRI view to use (sagittal, coronal, axial)
        slices_to_use: since most of the slices in a volume are not representable,
                       use only certain slices in a given range, e.g. (110, 180)
        mode: train or test
        tra
        """
        super(IXIdataset_Jigsaw, self).__init__()
        self.view = view
        self.slices_to_use = slices_to_use
        self.mode = mode
        self.image_transforms = image_transforms_
        self.tile_transforms = tile_transforms_
        self.permutations = permutations

        if self.view == 'coronal':
            self.slices_to_use = (90, 160)
        elif self.view == 'axial':
            self.slices_to_use = (110, 180)
        else:
            self.slices_to_use = slices_to_use

        if self.mode =='train':
            print("Start reading the training data list...")
        else:
            print("Start reading the test data list...")

        self.gtlists = natsorted(glob.glob(root + '*.nii'))

        self.idx1, self.idx2, self.idx3 = 0, 2, 1

        if self.view == 'axial':
            self.idx1, self.idx2, self.idx3 = 2, 1, 0
        elif view == 'coronal':
            self.idx1, self.idx2, self.idx3 = 1, 2, 0
        elif view == 'sagittal':
            self.idx1, self.idx2, self.idx3 = 0, 2, 1

        # Getting the shape of a single volume
        free_image_temp = nib.load(self.gtlists[0]).get_data() # 150*256*256
        free_image = free_image_temp.transpose(self.idx1, self.idx2, self.idx3) # 150*256*256
        self.c, self.h, self.w = free_image.shape # e.g. 150*256*256

        if self.mode =='train':
            print("End reading the training data list...")
        else:
            print("End reading the test data list...")

    def __len__(self):
        return len(self.gtlists)*(self.slices_to_use[1] - self.slices_to_use[0]) # num of images * num of slices

    def __getitem__(self, idx):
        subject_idx = idx // (self.slices_to_use[1] - self.slices_to_use[0])
        slice_idx = self.slices_to_use[0] + idx % (self.slices_to_use[1] - self.slices_to_use[0])

        free_image_temp = nib.load(self.gtlists[subject_idx]).get_data()
        free_image = free_image_temp.transpose(self.idx1, self.idx2, self.idx3)

        free_image = (free_image - free_image.min())/(free_image.max() - free_image.min())
        
        free_slice = free_image[slice_idx, ::-1, :]
        if self.view == 'sagittal':
            free_slice = free_image[slice_idx, ::-1, :]
        elif self.view == 'coronal':
            free_slice = free_image[slice_idx, ::-1, :]
        elif self.view == 'axial':
            free_slice = free_image[slice_idx, :, :]
        free_slice = torch.from_numpy(np.expand_dims(free_slice,0).copy())
        
        if self.image_transforms:
            free_slice = self.image_transforms(free_slice)
            # free_slice = free_slice.permute((2, 0, 1)).contiguous()
            # assert free_slice.shape[0] == 1, "1st dimension of an input is not 1"
        
        a = 75
        tiles = [None] * 9
        for n in range(9):
            i = n % 3
            j = n //3
            # c = [a*i*2+a,a*j*2+a]
            # tile = free_slice[:, int(c[1]-a) : int(c[1]+a+1), int(c[0]-a) : int(c[0]+a+1)]
            # import pdb; pdb.set_trace()
            tile = free_slice[:, a*i:a*(i+1)+1, a*j:a*(j+1)+1]
            tile = self.tile_transforms(torch.unsqueeze(tile, 1))
            tile = torch.squeeze(tile, 1)
            # Normalize the patches indipendently to avoid low level features shortcut
            #m = tile.mean()
            #s = tile.std()
            #norm = transforms.Normalize(mean=[m, m, m],
                                        #std =[s, s, s])
            #tile = norm(tile)
            tiles[n] = tile
    
        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data,0)
        return data, int(order)


class HarPdataset(Dataset):
    def __init__(self, dataroot='/home/guest/qasymjomart/seg_mri/ADNI/ADNI-final/',
                       labelsroot='/home/guest/qasymjomart/seg_mri/Released_data_NII_v1.3/woojin/',
                       metaroot='/home/guest/qasymjomart/seg_mri/woojin/harp_meta.txt',
                       indexes=None,
                       view = 'sagittal',
                       mode='train',
                       use_only_brain_slices = 0,
                       transforms_ = None):
        """
        This __init__ function initializes the class.
        dataroot: is the root to the nii files to be used in training or testing
        labelsroot: is the root to the nii files that contain labeled masks
        metaroot: is the root to the txt that tells
        view: is the MRI view to use (sagittal, coronal, axial)
        mode: train or test
        """
        super(HarPdataset, self).__init__()
        self.dataroot = dataroot
        self.labelsroot = labelsroot
        self.indexes = indexes
        self.view = view
        self.mode = mode
        self.use_only_brain_slices = use_only_brain_slices
        self.transforms = transforms_
        self.index_range = None

        if self.indexes is None:
            raise 'Please input meta data indexes that to be included into this dataset.'

        if self.mode =='train':
            print("Start reading the HarP training data list...")    
        else:
            print("Start reading the HarP test data list...")
        
        # self.meta = np.array(open(self.metaroot, 'r').readlines())[np.array(self.indexes)] # Convert to numpy because of indexing from index array

        self.gtlists_data = natsorted(glob.glob(self.dataroot + '*.nii'))
        self.gtlists_labels = natsorted(glob.glob(self.labelsroot + '*/*.nii'))

        if self.view == 'axial':
            self.idx1, self.idx2, self.idx3 = 2, 1, 0
        elif view == 'sagittal':
            self.idx1, self.idx2, self.idx3 = 0, 2, 1
        elif view == 'coronal':
            self.idx1, self.idx2, self.idx3 = 1, 2, 0

        # Getting the shape of a single volume
        free_image_temp = nib.load(self.gtlists_data[0]).get_data() # 
        free_image = free_image_temp.transpose(self.idx1, self.idx2, self.idx3) #
        self.c, self.h, self.w = free_image.shape #

        if self.mode =='train':
            print("End reading the training data list...")
        else:
            print("End reading the test data list...")

    def __len__(self):
        if self.use_only_brain_slices == 0:
            return len(self.indexes) * self.c # len of the meta file
        elif self.use_only_brain_slices == 1:
            return len(self.indexes) * (140 - 60)

    def __getitem__(self, idx):

        if self.use_only_brain_slices == 0:
            vol_idx = idx // self.c
            slice_idx = idx % self.c
        elif self.use_only_brain_slices == 1:
            vol_idx = idx // (140 - 60)
            slice_idx = idx % (140 - 60)

        data_file_name = self.indexes[vol_idx]
        metadata = data_file_name.split("/")[-1]
        
        mask_left_name  = [gtlist for gtlist in self.gtlists_labels if metadata[:15] in gtlist and 'L.nii' in gtlist][0]
        mask_right_name = [gtlist for gtlist in self.gtlists_labels if metadata[:15] in gtlist and 'R.nii' in gtlist][0]
        
        free_data = nib.load(data_file_name).get_data()
        free_data = free_data.transpose(self.idx1, self.idx2, self.idx3)

        free_data = (free_data - free_data.min())/(free_data.max() - free_data.min())

        # Get masks
        free_mask_left = nib.load(mask_left_name).get_data()
        free_mask_left = free_mask_left.transpose(self.idx1, self.idx2, self.idx3)

        free_mask_right = nib.load(mask_right_name).get_data()
        free_mask_right = free_mask_right.transpose(self.idx1, self.idx2, self.idx3)

        free_mask = free_mask_left + free_mask_right

        free_slice = free_data[slice_idx, ::-1, ::-1]
        free_mask_slice = free_mask[slice_idx, ::-1, ::-1]

        free_slice = torch.from_numpy(np.expand_dims(free_slice,0).copy())
        free_mask_slice = torch.from_numpy(np.expand_dims(free_mask_slice,0).copy()).type(torch.FloatTensor)
        
        if self.transforms:
            free_slice = self.transforms(free_slice)
            free_mask_slice = self.transforms(free_mask_slice)
        
        return free_slice, free_mask_slice

class Decathlon(Dataset):
    def __init__(self, dataroot='/home/guest/qasymjomart/seg_mri/Task04_Hippocampus/imagesTr/',
                       labelsroot='/home/guest/qasymjomart/seg_mri/Task04_Hippocampus/labelsTr/',
                       metaroot='/home/guest/qasymjomart/seg_mri/Task04_Hippocampus/',
                       indexes=None,
                       view = 'sagittal',
                       mode='train',
                       transforms_ = None):
        """
        This __init__ function initializes the class.
        dataroot: is the root to the nii files to be used in training or testing
        labelsroot: is the root to the nii files that contain labeled masks
        metaroot: is the root to the txt that tells
        view: is the MRI view to use (sagittal, coronal, axial)
        mode: train or test
        """
        super(Decathlon, self).__init__()
        self.dataroot = dataroot
        self.labelsroot = labelsroot
        self.metaroot = metaroot
        self.indexes = indexes
        self.view = view
        self.mode = mode
        self.transforms = transforms_

        if self.indexes is None:
            raise 'Please input meta data indexes that to be included into this dataset.'

        if self.mode =='train':
            print("Start reading the Decathlon training data list...")    
        else:
            print("Start reading the Decathlon test data list...")
        
        self.gtlists_data = natsorted(glob.glob(self.dataroot + '*.nii.gz'))
        self.gtlists_labels = natsorted(glob.glob(self.labelsroot + '*.nii.gz'))

        self.gtlists_data = [ e for e in self.indexes if e in self.gtlists_data ]

        if self.view == 'axial': # NEED TO BE REVISED
            self.idx1, self.idx2, self.idx3 = 2, 0, 1
        elif view == 'sagittal':
            self.idx1, self.idx2, self.idx3 = 1, 0, 2
        elif view == 'coronal':
            self.idx1, self.idx2, self.idx3 = 0, 1, 2

        num_a, num_b = 0, 0
        self.dict_cs = {}
        # Getting the shape of each volume
        self.total_c = 0
        for ii in range(len(self.gtlists_data)):
            free_image_temp = nib.load(self.gtlists_data[ii]).get_data() # 
            free_image = free_image_temp.transpose(self.idx1, self.idx2, self.idx3) #
            self.c, self.h, self.w = free_image.shape #
            self.total_c += self.c
            num_b += self.c
            self.dict_cs[self.gtlists_data[ii]] = range(num_a, num_b)
            num_a = num_b

        if self.mode =='train':
            print("End reading the training data list...")
        else:
            print("End reading the test data list...")

    def __len__(self):
        return self.total_c # len of the meta file

    def __getitem__(self, idx):
        for vol_name in self.dict_cs.keys():
            if idx in self.dict_cs[vol_name]:
                vol_idx_path = vol_name
                slice_idx = idx - self.dict_cs[vol_name][0]

        metadata = vol_idx_path.split('/')[-1]
        # data_file_name = [gtlist for gtlist in self.gtlists_data if metadata in gtlist][0]
        mask_name  = [gtlist for gtlist in self.gtlists_labels if metadata in gtlist][0]
        # slice_idx = int(metadata[1])

        free_data = nib.load(vol_idx_path).get_data()
        free_data = free_data.transpose(self.idx1, self.idx2, self.idx3)

        free_data = (free_data - free_data.min())/(free_data.max() - free_data.min())

        # Get masks
        free_mask = nib.load(mask_name).get_data()
        free_mask = free_mask.transpose(self.idx1, self.idx2, self.idx3)

        assert free_data.shape == free_mask.shape

        free_slice = free_data[slice_idx, :, :]
        free_mask_slice = free_mask[slice_idx, :, :]
        
        # Replace labels for anterior hippocampus that are 2 with 1 (mix with posterior hippocampus)
        free_mask_slice = np.where(free_mask_slice == 2, 1, free_mask_slice)

        free_slice = torch.from_numpy(np.expand_dims(free_slice,0).copy()).type(torch.FloatTensor)
        free_mask_slice = torch.from_numpy(np.expand_dims(free_mask_slice,0).copy()).type(torch.FloatTensor)

        # temp_background = torch.zeros(free_mask_slice.shape)
        # temp_background[free_mask_slice == 0.0] = 1.0
        # free_mask_slice = torch.cat((free_mask_slice, temp_background.type(torch.FloatTensor)), dim=0)
        
        if self.transforms:
            free_slice = self.transforms(free_slice)
            free_mask_slice = self.transforms(free_mask_slice)
        
        # import pdb; pdb.set_trace()

        return free_slice, free_mask_slice
