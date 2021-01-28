# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:25:40 2018

@author: kev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4

class ConvBnRelu2d(nn.Module):
    """
        This class composes  2D convolution layer, batch normalization layer and ReLU into one.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
                 is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn is False: self.bn = None
        if is_relu is False: self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


## original 3x3 stack filters used in UNet
class StackEncoder(nn.Module):
    """
        Encoding block consists from two ConvBnRelu2d layers and pooling layer
    """
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(nn.Module):
    """
        Decoding block consists from two ConvBnRelu2d layers and upsample layer
    """
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding,
                         dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y
        
class UNet96cls(nn.Module):
    def __init__(self, in_shape, out_channels=3):
        """
            Implementation of UNet 
            https://arxiv.org/abs/1505.04597
            Size of output tensor (w * h) is equal to input image size.
            
            Args:
                in_shape: shape of input image, only channels value of image used
                out_channels: number of cannels in output tensor
        """
        super(UNet96cls, self).__init__()
        C, H, W = in_shape

        self.down1 = StackEncoder(C, 32, kernel_size=3)
        self.down2 = StackEncoder(32, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 192, kernel_size=3)
        self.down5 = StackEncoder(192, 288, kernel_size=3)
        self.down6 = StackEncoder(288, 432, kernel_size=3)

        self.center = nn.Sequential(
            ConvBnRelu2d(432, 768, kernel_size=3, padding=1, stride=1),
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1),
        )
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 432, 432, kernel_size=3)
        self.up5 = StackDecoder(432, 288, 288, kernel_size=3)
        self.up4 = StackDecoder(288, 192, 192, kernel_size=3)
        self.up3 = StackDecoder(192, 128, 128, kernel_size=3)
        self.up2 = StackDecoder(128, 64, 64, kernel_size=3)
        self.up1 = StackDecoder(64, 32, 32, kernel_size=3)
        self.classify = nn.Conv2d(32, out_channels, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        
        out = self.center(out)
        
        out = self.up6(down6, out)        
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out      

