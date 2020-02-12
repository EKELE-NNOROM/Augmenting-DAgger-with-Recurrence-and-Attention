import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from utils import weights_init_xavierUniform as weights_init
import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Recurrent_block_old(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.conv = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn = nn.ModuleList([nn.BatchNorm2d(ch_out) for i in range(t)])
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        rx = x
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x) + self.shortcut(rx)
            x = self.relu(x1)
            x = self.bn[i](x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Conv_block_2(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, k1, k2, s1, s2, p1, p2):
        super(Conv_block_2, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=k1, stride=s1, padding=p1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ch_mid, ch_out, kernel_size=k2, stride=s2, padding=p2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(ch_mid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.apply(weights_init)

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.bn(x1)
        x2 = self.conv_2(x1)
        # print('Working now')
        # print('x2',x2)
        return x2

class Conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_mid_2, ch_out, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(Conv_block_3, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=k1, stride=s1, padding=p1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_mid_2, kernel_size=k2, stride=s2, padding=p2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ch_mid_2, ch_out, kernel_size=k3, stride=s3, padding=p3, bias=True),
            nn.ReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(ch_mid_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.apply(weights_init)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.bn(x1)
        x3 = self.conv_2(x2)
        # print('Working now')
        # print('x3',x3)
        return x3
