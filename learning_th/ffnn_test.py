#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   feed forward neural network
@File   :   ffnn_test.py
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/11/22 5:39 PM   yinzikang      1.0         None
"""
import torch as th
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size[0]),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size[-1], output_size), nn.Sigmoid())

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)

        return output


if __name__ == "__main__":
    # device
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    writer = SummaryWriter('ffnn_test_tensorboard')

    # network
    ins = 2
    hds = [2]
    ous = 1
    net1 = FeedForwardNetwork(ins, hds, ous).to(device)
    print(net1)
    print(sum(p.numel() for p in net1.parameters() if p.requires_grad))
    for layer_tensor_name, tensor in net1.named_parameters():
        print('Layer {}: {} elements'.format(layer_tensor_name, th.numel(tensor)))

    # Todo: optimizer

    # 全链接层对输入的要求只有最后一个
    test_input1 = th.Tensor([[[[[[1, 1], [2, 2]]]]]]).to(device)
    test_input2 = th.Tensor([[1, 1], [2, 2]]).to(device)
    test_input3 = th.Tensor([1, 1]).to(device)
    print(test_input1)
    print(test_input1.device)

    test_output1 = net1(test_input1)
    # print(test_output1)
    # print(test_output1.device)
    # test_output2 = net1(test_input2)
    # print(test_output2)
    # print(test_output1.device)
    # test_output3 = net1(test_input3)
    # print(test_output3)
    # print(test_output3.device)

    # vis = make_dot(test_output1, params=dict(net1.named_parameters())).render("ffnn", format="png")

    writer.add_graph(net1, test_input3)
    writer.close()
