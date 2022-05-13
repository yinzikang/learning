#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   
@File   :   spinnup_vpg.test.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/27/22 12:30 PM   yinzikang      1.0         None
"""

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from spinup.algos.pytorch.vpg.vpg import vpg
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
from spinup.utils.run_utils import setup_logger_kwargs

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--exp_name', type=str, default='222')
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

vpg(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs)
