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
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


class PolicyNetwork(nn.Module):
    """the Network to output action approximate state value

    input goal, state and action, outputs value of the state
    """

    def __init__(self, goal_dimension, state_dimension, action_dimension):
        super(PolicyNetwork, self).__init__()
        self.ff_branch = nn.Sequential(nn.Linear(in_features=goal_dimension + state_dimension,
                                                 out_features=128), nn.ReLU())
        self.lstm_branch = nn.Sequential(nn.Linear(in_features=state_dimension + action_dimension,
                                                   out_features=128), nn.ReLU(),
                                         nn.LSTM(input_size=128, hidden_size=128))
        self.fc_part = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU(),
                                     nn.Linear(in_features=128, out_features=128))

    def forward(self, parameter, goal, state_series, action_series):
        ff_out = self.ff_branch(parameter, goal, action_series, state_series)
        lstm_out = self.lstm_branch(state_series, action_series)
        action = self.fc_part(th.cat((ff_out, lstm_out)))

        return action


class ValueNetwork(nn.Module):
    """the Network to approximate state value

    input goal, state and action, outputs value of the state
    """

    def __init__(self, parameter_dimension, goal_dimension, state_dimension, action_dimension):
        super(ValueNetwork, self).__init__()
        self.ff_branch = nn.Sequential(nn.Linear(in_features=parameter_dimension + goal_dimension +
                                                             state_dimension + action_dimension,
                                                 out_features=128), nn.ReLU())
        self.lstm_branch = nn.Sequential(nn.Linear(in_features=state_dimension + action_dimension,
                                                   out_features=128), nn.ReLU(),
                                         nn.LSTM(input_size=128, hidden_size=128))
        self.fc_part = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU(),
                                     nn.Linear(in_features=128, out_features=128), nn.ReLU())

    def forward(self, goal, state_series, action_series):
        ff_out = self.ff_branch(goal, action_series, state_series)
        lstm_out = self.lstm_branch(state_series, action_series)
        out = self.fc_part(th.cat((ff_out, lstm_out)))

        return out


if __name__ == "__main__":
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    writer = SummaryWriter('lstm_test_tensorboard')

    para_dims = 10
    goal_dims = 1
    state_dims = 52
    action_dims = 7
    actor_network = PolicyNetwork(parameter_dimension=para_dims,
                                  goal_dimension=goal_dims,
                                  state_dimension=state_dims,
                                  action_dimension=action_dims)
    critic_network = ValueNetwork(goal_dimension=goal_dims,
                                  state_dimension=state_dims,
                                  action_dimension=action_dims)
    print(actor_network)
    print(sum(p.numel() for p in actor_network.parameters() if p.requires_grad))
    for layer_tensor_name, tensor in actor_network.named_parameters():
        print('Layer {}: {} elements'.format(layer_tensor_name, th.numel(tensor)))
    print(critic_network)
    print(sum(p.numel() for p in critic_network.parameters() if p.requires_grad))
    for layer_tensor_name, tensor in critic_network.named_parameters():
        print('Layer {}: {} elements'.format(layer_tensor_name, th.numel(tensor)))

    writer.add_graph(actor_network, test_input3)
    writer.add_graph(critic_network, test_input3)
    writer.close()
