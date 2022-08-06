# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: qasymjomart
Based also on:
https://github.com/milesial/Pytorch-UNet
and

"""

from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('Utils')

def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class JigsawUNetDown(nn.Module):
    """
    This is an UNet encoder-based architecture for training of Jigsaw task
    
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(JigsawUNetDown, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.down = nn.Sequential()
        self.down.add_module('double_conv', DoubleConv(n_channels, 64))
        self.down.add_module('down1', Down(64, 128))
        self.down.add_module('down2', Down(128, 256))
        self.down.add_module('down3', Down(256, 512))
        self.down.add_module('down4', Down(512, 1024 // factor))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear((1024//factor)*16, 1024))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, n_classes))

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.down(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.down = nn.Sequential()
        self.down.add_module('double_conv', DoubleConv(n_channels, 64))
        self.down.add_module('down1', Down(64, 128))
        self.down.add_module('down2', Down(128, 256))
        self.down.add_module('down3', Down(256, 512))
        self.down.add_module('down4', Down(512, 1024 // factor))

        self.up = nn.Sequential()
        self.up.add_module('up1', Up(1024, 512 // factor, bilinear))
        self.up.add_module('up2', Up(512, 256 // factor, bilinear))
        self.up.add_module('up3', Up(256, 128 // factor, bilinear))
        self.up.add_module('up4', Up(128, 64))

        self.out_conv = OutConv(64, n_classes)

        self.apply(weights_init)

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        x_down = [x]
        for layer in self.down:
            x_down.append(layer(x_down[-1]))
        
        x_up = [x_down[-1]]
        ii = 1
        for layer in self.up:
            x_up.append(layer(x_up[-1], x_down[-1-ii]))
            ii += 1

        logits = self.out_conv(x_up[-1])
        
        return logits


""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        t = 0.5 # threshold
        # return Variable((torch.sigmoid(self.conv(x)) > t).float(), requires_grad=True)
        return torch.sigmoid(self.conv(x))
    