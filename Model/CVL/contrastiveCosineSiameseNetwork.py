#!/usr/bin/env python
# coding: utf-8



import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow.keras.backend as K



class ContrastiveCosineSiamese(nn.Module):
    """
    A reimplementation of the original paper
    """
    def __init__(self, input_channels=1):
        super(ContrastiveCosineSiamese,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,10),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,128,4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,4),
            nn.ReLU(),
        )
        
        self.linear = nn.Sequential(nn.Linear(30976,4096), nn.Sigmoid())
        #self.out = nn.Linear(4096,1)
        
    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1,out2



