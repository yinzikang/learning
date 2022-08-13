#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""测试高维张量乘法

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
8/11/22 7:23 PM   yinzikang      1.0         None
"""
import torch

true_traj = torch.randn(32, 2, 6)
newton_pred = torch.zeros_like(true_traj)
newton_pred1 = torch.zeros_like(true_traj)
newton_pred2 = torch.zeros_like(true_traj)
lstm_pred = torch.randn_like(true_traj)
output1 = torch.zeros_like(true_traj)
output2 = torch.zeros_like(true_traj)

g = -9.81
dt = 1 / 120.
batch_size = 32
# kalman初始化，一条轨迹一个P，ABHQR共用
A = torch.tensor([[1, 0, 0, dt, 0, 0],
                   [0, 1, 0, 0, dt, 0],
                   [0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
B = torch.tensor([0, 0, 0, 0, 0, dt ** 2 / 2 * g])
H = torch.eye(6)
P = 0.0001 * torch.eye(6).unsqueeze(0).repeat(batch_size, 1, 1)
P1 = torch.zeros_like(P)
P2 = torch.zeros_like(P)
R = 0.1 * torch.eye(6)
Q = 0.01 * torch.eye(6)
I = torch.eye(6)
# eq1

for traj_idx in range(batch_size):  # eq1
    newton_pred[traj_idx, -1, :] = A @ true_traj[traj_idx, -1, :] + B
newton_pred1[:, -1, :] = true_traj[:, -1, :] @ A.T + B
newton_pred2[:, -1, :] = torch.matmul(true_traj[:, -1, :], A.T) + B

print(newton_pred.equal(newton_pred2))
print(newton_pred1.equal(newton_pred2))

for traj_idx in range(batch_size):
    P1[traj_idx] = A @ P[traj_idx] @ A.T + Q  # eq2
    K1 = P1[traj_idx] @ H.T @ (H @ P1[traj_idx] @ H.T + R).inverse()
    output1[traj_idx, -1, :] = newton_pred[traj_idx, -1, :] + \
                                    K1 @ (lstm_pred[traj_idx, -1, :] -
                                         H @ newton_pred[traj_idx, -1, :])
    P1[traj_idx] = (I - K1 @ H) @ P1[traj_idx]
P2 = A @ P @ A.T + Q
K2 = P2 @ H.T @ (H @ P2 @ H.T + R).inverse()
output2[:, -1, :] = newton_pred[:, -1, :] + (K2 @ (lstm_pred[:, -1, :] - newton_pred[:, -1, :] @ H.T).unsqueeze(-1)).reshape(batch_size,-1)
P2 = (I - K2 @ H) @ P2

print(P1.equal(P2))
print(K1.equal(K2[-1]))
print(output1.equal(output2))
