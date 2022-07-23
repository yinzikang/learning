#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""learn how to use tensorboard with pytorch

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/17/22 8:13 PM   yinzikang      1.0         None
"""

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i ** 2, global_step=i)
    writer.add_scalar('exponential', 2 ** i, global_step=i)
    writer.close()
