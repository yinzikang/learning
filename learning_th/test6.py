#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
8/11/22 8:35 PM   yinzikang      1.0         None
"""
import torch

A = torch.tensor([[0, 2, 3, 4], [1, 24, 3, 4], [1, 25, 3, 4], [1, 2, 3, 4]])
B = torch.tensor([[-1, 23, 9, 4], [5, 6, 7, 78]])  # batch=2,dim=4
ans1 = torch.zeros_like(B)
for traj_idx in range(2):  # eq1
    ans1[traj_idx, :] = A @ B[traj_idx, :]
ans2 = torch.matmul(B, A.T)
print(ans1)
print(ans2)
