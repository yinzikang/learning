#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
7/24/22 2:45 PM   yinzikang      1.0         None
"""

import torch
import torch.nn as nn

loss_func = nn.MSELoss()
a = torch.tensor([0., 0., 4.])
b = torch.tensor([1., 6., 1.])
print(loss_func(a, b))
print(((a.numpy()-b.numpy()) ** 2).mean())