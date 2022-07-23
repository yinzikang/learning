#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
7/1/22 5:26 PM   yinzikang      1.0         None
"""
import torch
import torch.nn as nn

FEATURE_DIMENSION = 18  # encoder与decoder输入特征维度
NUM_HEAD = 3
NUM_ENCODER_BLOCK = 1
NUM_DECODER_BLOCK = 1
DIM_FORWARD = 128  # encoder与decoder中全链接网络维度

net = nn.Transformer(d_model=FEATURE_DIMENSION, nhead=NUM_HEAD,
                     num_decoder_layers=NUM_ENCODER_BLOCK, num_encoder_layers=NUM_DECODER_BLOCK,
                     dim_feedforward=DIM_FORWARD)


# if __name__ == "__main__":
#     pass
