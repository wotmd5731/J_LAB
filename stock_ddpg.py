
# coding: utf-8

# In[1]:


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# In[2]:


from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Use CUDA</h2>

# In[3]:


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# <h2>Replay Buffer</h2>

# In[5]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


# <h2>Normalize action space</h2>

# In[8]:


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action


# <h2>Ornstein-Uhlenbeck process</h2>
# Adding time-correlated noise to the actions taken by the deterministic policy<br>
# <a href="https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process">wiki</a>

# In[12]:


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.low          = 0
        self.high         = 1
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    
#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


# In[16]:


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


# <h1> Continuous control with deep reinforcement learning</h1>
# <h2><a href="https://arxiv.org/abs/1509.02971">Arxiv</a></h2>

# In[18]:


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x
    
    def get_action(self, state):
#        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state.to(device))
        return action.detach().cpu().numpy()[0, 0]


# <h2>DDPG Update</h2>

# In[19]:


def ddpg_update(batch_size, 
           gamma = 0.99,
           min_value=-np.inf,
           max_value=np.inf,
           soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = torch.FloatTensor(state).reshape((batch_size,-1)).to(device)
    next_state = torch.FloatTensor(next_state).reshape((batch_size,-1)).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


    return policy_loss.detach().cpu(),value_loss.detach().cpu()


# In[]:
            
            
import logging
import os
import settings
import data_manager
import pandas as pd
import numpy as np            
import random

class env_stock():
    def __init__(self,init_money,quantize):
        
        stock_code = '005930'
        self.start_date = '2010-03-01'
        self.end_date = '2011-03-04'

        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         'data/chart_data/{}.csv'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)
        
        # 기간 필터링
        training_data = training_data[(training_data['date'] >= self.start_date) &
                                      (training_data['date'] <= self.end_date)]
        training_data = training_data.dropna()
        
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]


        chart_data['date']= pd.to_datetime(chart_data.date).astype(np.int64)/1000000
        data = torch.from_numpy(chart_data.values)

        self.data = torch.stack([data[:,0],data[:,4],data[:,5]],dim=1).float()

        self.data = self.data - self.data.mean(dim=0)
        self.data = self.data/self.data.std(0)

        self.count_max= self.data.size(0)
        self.init_money = init_money
        self.quantize = quantize
        
    def reset(self):
        self.prev_action = 0
        self.count = 0
        self.pocket= self.init_money
        return self.data[self.count].view(1,-1)
        pass
        

    def step(self,action):
        #continuouse action space
        action = action.item()
        a_t = round((self.prev_action - action)*self.quantize)
        r_t = d_t= 0
        
        cost = self.data[self.count,1]*a_t*0.99
        self.pocket += cost
        
        self.prev_action = action
        r_t = cost
        if self.count+1 == self.count_max:
#            self.pocket += self.data[self.count,1]*(self.prev_action*quantize)
            d_t = 1
            r_t = self.pocket 

        else:
            self.count +=1

        return self.data[self.count].view(1,-1),r_t ,d_t,0
        pass
    
    def vis(self):
        vis.line(Y=self.data[:,1],name='price')
        vis.line(Y=self.data[:,2],name='vol')
        

env = env_stock(0,100)
env.reset()




# In[25]:







#env = NormalizedActions(gym.make("Pendulum-v0"))
ou_noise = OUNoise(1)

state_dim  = 3
action_dim = 1
hidden_dim = 256

value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)
    
    
value_lr  = 1e-4
policy_lr = 1e-5

value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


# In[28]:


max_frames  = 12000
max_steps   = env.count_max
frame_idx   = 0
rewards     = []
batch_size  = 128


# In[29]:
import visdom

vis = visdom.Visdom()
win_p = vis.line(Y=torch.Tensor([0]), name='ploss')
win_v = vis.line(Y=torch.Tensor([0]), name='vloss')
win_r = vis.line(Y=torch.Tensor([0]), name='reward')
win_a = vis.line(Y=torch.Tensor([0]), name='action')

print(max_steps)
env.vis()
traj = []

while frame_idx < max_frames:
    state = env.reset()
    ou_noise.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        traj.append(action.item())
        next_state, reward, done, _ = env.step(action)
        
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            ploss,vloss=ddpg_update(batch_size)
            win_p = vis.line(X=torch.Tensor([frame_idx]), Y=ploss.view(1) , win = win_p, update='append')
            win_v = vis.line(X=torch.Tensor([frame_idx]), Y=vloss.view(1) , win = win_v,  update='append')
            
        state = next_state
        episode_reward += reward
        frame_idx += 1
        
#        if frame_idx % max(1000, max_steps + 1) == 0:
#            plot(frame_idx, rewards)
            
        if done:
            win_r = vis.line(X=torch.Tensor([frame_idx]), Y=torch.Tensor([episode_reward]).view(1) , win = win_r , update='append')
            win_a = vis.line(Y=torch.Tensor(traj) , win = win_a)
            
            break
    
    rewards.append(episode_reward)

