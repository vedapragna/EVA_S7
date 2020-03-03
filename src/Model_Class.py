# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:50:15 2020

@author: 11028434
"""

import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
            ,nn.BatchNorm2d(32)
            #,nn.Dropout(dropout_value)
        ) # output_size = 32, RF = 3


        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),dilation= 2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 30,  RF = 7

        # TRANSITION BLOCK 1
        self.convblock2_t = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
        ) # output_size = 32,  RF = 7
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16,  RF = 8


        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=1, groups = 32, bias=False), ## output_size = 16,  RF = 12
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=1, bias=False) ## output_size = 18,  RF = 12
            ,nn.ReLU()         
            ,nn.BatchNorm2d(64)
            ,nn.Dropout(dropout_value)
        ) # output_size = 18,  RF = 12
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 9, RF = 14

        # CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 9, RF = 22
        self.convblock4_t = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
        ) # output_size = 11, RF = 22
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 5, RF = 26


        # OUTPUT BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )# output_size = , RF = 42
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1, RF = 74

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF = 74


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)

        x = self.convblock2(x)
        x = self.convblock2_t(x)
        x = self.pool1(x)

        x = self.convblock3(x)
        x = self.pool2(x)
        #print(x.shape)
        
        #print(x.shape)
        x = self.convblock4(x)
        x = self.convblock4_t(x)
        x = self.pool3(x)

        x = self.convblock6(x)
        x = self.gap(x)        
        x = self.convblock5(x)

        x = x.view(-1, 10)
        return x    #F.log_softmax(x, dim=-1)

    def model_test(self):
        print("In Model Class")