#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   to try some gym envs
@File   :   gym_test_1.py
@author :   yinzikang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
3/24/22 7:41 PM   yinzikang      1.0         None
"""
import numpy as np
import gym
import os

os.system("unset LD_PRELOAD")
os.system('ls')

env = gym.make('FetchReach-v1')
obs = env.reset()
done = False


def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()


while not done:
    action = policy(obs['observation'], obs['desired_goal'])
    obs, reward, done, info = env.step(action)

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))

    env.render(mode="human")
