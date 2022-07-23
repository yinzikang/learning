#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   convolution neural network
@File   :   lenet_test.py
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/12/22 11:10 AM   yinzikang      1.0         None
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
        self.fc1 = nn.Sequential(nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(in_features=120, out_features=84), nn.Sigmoid())
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        # 分成(batch_size,-1)，送入全链接，相当于全连接的神经元个数只与单张图的卷积结果相关
        output = self.fc1(output.view(input.shape[0],-1))
        output = self.fc2(output)
        output = self.fc3(output)

        return output


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == "__main__":
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    writer = SummaryWriter('cnn_test_tensorboard')

    net1 = ConvolutionNetwork().to(device)
    # net1 = LeNet().to(device)
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

    # raw_img = plt.imread('lenna.jpeg')
    # fit_img = cv2.cvtColor(cv2.resize(raw_img, (32, 32)), cv2.COLOR_BGR2GRAY)
    # test_input1 = th.Tensor(fit_img).unsqueeze(0).unsqueeze(0).to(device)
    # test_output1 = net1(test_input1)
    # vis = make_dot(test_output1, params=dict(net1.named_parameters())).render("le_net", format="png")

    img_PIL = Image.open('lenna.jpeg')
    test_input1 = my_trans(img_PIL).unsqueeze(0).to(device)
    test_output1 = net1(test_input1)

    # plt.ion()
    # plt.imshow(test_input1)
    # plt.pause(3)
    # plt.close()

    writer.add_image(tag='input_image', img_tensor=np.array(img_PIL), global_step=1, dataformats='HWC')
    writer.add_graph(net1, test_input1)
    writer.close()
