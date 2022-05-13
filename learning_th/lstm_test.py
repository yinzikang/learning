#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""try to implement the LSTM neural network

network architecture from Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
two network: v-net and p-net
multi-branch,LSTM

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/13/22 8:16 PM   yinzikang      1.0         None
"""
import torch as th
import torch.nn as nn


class ValueNetwork(nn.Module):
    """the Network to approximate state value

    input goal, state and action, outputs value of the state
    """

    def __int__(self, goal_dimension, state_dimension, action_dimension):
        super(ValueNetwork, self).__int__()
        self.feed_forward_branch = nn.Linear(in_features=goal_dimension + state_dimension, out_features=128)
        self.recurrent_branch = nn.Sequential(nn.Linear(in_features=goal_dimension + state_dimension, out_features=128),
                                              nn.LSTM(input_size=state_dimension + action_dimension, hidden_size=128))


if __name__ == "__main__":
    pass
