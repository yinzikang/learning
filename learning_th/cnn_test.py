#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/19/22 10:51 AM   yinzikang      1.0         None
"""

import numpy as np
import torch as th
import torch.nn as nn
from torchviz import make_dot
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
                                   nn.Sigmoid())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
                                   nn.Sigmoid())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        output = self.conv1(input)
        # output = self.pool1(output)
        output = self.conv2(output)
        # output = self.pool2(output)
        return output


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
writer = SummaryWriter('cnn_test_tensorboard')

net1 = ConvolutionNetwork().to(device)
print(net1)
print(sum(p.numel() for p in net1.parameters() if p.requires_grad))
for layer_tensor_name, tensor in net1.named_parameters():
    print('Layer {}: {} elements'.format(layer_tensor_name, th.numel(tensor)))
# print(list(net1.parameters()))

my_trans = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

img_PIL = Image.open('lenna.jpeg')
test_input1 = my_trans(img_PIL).unsqueeze(0).to(device)
test_output1 = net1(test_input1)

writer.add_image(tag='input_image', img_tensor=np.array(img_PIL), global_step=1, dataformats='HWC')
writer.add_graph(net1, test_input1)
writer.close()
