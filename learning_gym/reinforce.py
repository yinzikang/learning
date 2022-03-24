import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_test

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = gym_test.Car2DEnv()
env = gym.make('CartPole-v0')
env.reset()
env.render()
# env.close()


Hidden = 9
GAMMA = 0.9
LR = 0.03

EPISODE = 500
STEP = 100
TEST = 1


class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        """
        神经网络初始化，调用Net的父类nn.Module的初始化
        :param obs_dim: 把所有的观测作为网络输入
        :param act_dim: 把所有的输出作为网络输出
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, Hidden)
        self.fc2 = nn.Linear(Hidden, act_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class RF(object):
    def __init__(self, env):
        # 环境的状态和动作维度
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        # 存放每个episode的s,a,r
        self.ep_obs, self.ep_act, self.ep_r = [], [], []
        # 初始化神经网络
        self.net = Net(obs_dim=self.obs_dim, act_dim=self.act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.time_step = 0

    # 选择动作的函数
    def choose_action(self, obs):
        obs = torch.FloatTensor(obs).to(device)  # 转换为torch的格式
        action = self.net.forward(obs)
        with torch.no_grad():  # 不进行参数的更新
            action = F.softmax(action, dim=0).cuda().data.cpu().numpy()
        action = np.random.choice(range(action.shape[0]), p=action)  # 根据softmax输出的概率来选择动作

        return action

    # 存储一个episode的状态、动作和回报的函数
    def store_transition(self, obs, act, r):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_r.append(r)

    # 更新策略网络的函数
    def learn(self):
        self.time_step += 1  # 记录走过的step
        # 记录Gt的值
        discounted_ep_rs = np.zeros_like(self.ep_r)
        running_add = 0
        # 计算未来总收益
        for t in reversed(range(0, len(self.ep_r))):  # 反向计算
            running_add = running_add * GAMMA + self.ep_r[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # 输出网络计算出的每个动作的概率值
        act_prob = self.net.forward(torch.FloatTensor(self.ep_obs).to(device))
        # 进行交叉熵的运算
        neg_log_prob = F.cross_entropy(input=act_prob, target=torch.LongTensor(self.ep_act).to(device),
                                       reduction='none')
        # 计算loss
        loss = torch.mean(neg_log_prob * discounted_ep_rs)

        # 反向传播优化网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习后清空s,r,a的数组
        self.ep_r, self.ep_act, self.ep_obs = [], [], []


def t_episode(test_env, agent):
    total_reward = 0
    for episode in range(TEST):
        obs = test_env.reset()
        for step in range(STEP):
            test_env.render()
            act = agent.choose_action(obs)
            next_obs, reward, done, _ = test_env.step(act)

            obs = next_obs
            total_reward += reward

            if done:
                break
    return total_reward / TEST  # 计算测试的平均reward


if __name__ == "__main__":
    # 初始化RF类
    agent = RF(env)

    # 进行训练
    for episode in range(EPISODE):
        obs = env.reset()
        for step in range(STEP):
            # 与环境的交互
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            # 存储一个episode中每个step的s,a,r
            agent.store_transition(obs, action, reward)
            # 进入下一个状态
            obs = next_obs
            # 每个episode结束再进行训练(MC)
            if done:
                agent.learn()
                break
        # 每100个episode进行测试
        if episode % 100 == 0:
            avg_reward = t_episode(env, agent)
            print('Episode: ', episode, 'Test_reward: ', avg_reward)
    env.close()
