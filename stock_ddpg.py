
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
import visdom

vis = visdom.Visdom()
vis.close()
# In[2]:


from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Use CUDA</h2>

# In[3]:


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

# <h2>Replay Buffer</h2>

# In[5]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
#    def push(self,mem):
#        if len(self.buffer) < self.capacity:
#            self.buffer.append(None)
#        self.buffer[self.position] = mem
#        self.position = (self.position + 1) % self.capacity
#    
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size,seq_size=1):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
#        batch_data = random.sample(self.buffer,batch_size)
#        state, action, reward, next_state, done = [],[],[],[],[]
#        for data in batch_data:
#            start = random.randint(0,len(data)-seq_size)
#            aa = data[start:start+seq_size]
#            t0,t1,t2,t3,t4 = list(zip(*aa))
#            
#            state.append(torch.stack(t0,dim=1))
#            action.append(torch.stack(t1,dim=1))
#            reward.append(torch.stack(t2,dim=0))
#            next_state.append(torch.stack(t3,dim=1))
#            done.append(torch.tensor(t4))
#        
#        state  = torch.cat(state).reshape((batch_size,seq_size,-1)).to(device)
#        action = torch.cat(action).reshape((batch_size,seq_size,-1)).to(device)
#        reward = torch.cat(reward).reshape((batch_size,seq_size,-1)).to(device)
#        next_state = torch.cat(next_state).reshape((batch_size,seq_size,-1)).to(device)
#        done = torch.cat(done).reshape((batch_size,seq_size,-1)).to(device)
#        return state, action, reward, next_state, done
    
    
    
    def __len__(self):
        return len(self.buffer)



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
        self.low          = -1
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
        ou_state = torch.from_numpy(self.evolve_state()).float()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.clamp(action + ou_state, self.low, self.high)
    
#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


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
        x = torch.tanh(self.linear3(x))
        return x
    
    def get_action(self, state):
#        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state.to(device))
        return action.detach().cpu()


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
    action     = torch.FloatTensor(action).reshape((batch_size,-1)).to(device)
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
            
            
import os
import data_manager
import pandas as pd
import numpy as np            
import random
import datetime

class env_stock():
    def __init__(self,count_max,view_seq,init_money):
        self.count_max= count_max
        self.view_seq = view_seq
        self.win1= vis.line(Y=torch.Tensor([0]))
        self.win2= vis.line(Y=torch.Tensor([0]))
        self.init_money = init_money
        
    def reset(self):
        stock_code = np.random.choice(['005930' , '000270','000660','005380','005490','009240','009540'])
        
        self.prev_action = 0
        self.count = 0
        self.balance = self.init_money
        self.num_stocks = 0
        self.sum_action = 0
        
        chart_data = data_manager.load_chart_data(
            os.path.join('./',
                         'data/chart_data/{}.csv'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)
        
        # 기간 필터링
        start = random.randint(self.view_seq ,(len(training_data)-self.count_max-200))
        
        training_data = training_data[start-self.view_seq :start+self.count_max+200]
        
#        training_data = training_data[(training_data['date'] >= self.start_date) &
#                                      (training_data['date'] <= self.end_date)]
        training_data = training_data.dropna()
        
        
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.chart_data = training_data[features_chart_data]
        self.chart_data['date']= pd.to_datetime(self.chart_data.date).astype(np.int64)/1e12


        
        # 학습 데이터 분리
        features_training_data = [
            'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio',
            'close_ma5_ratio', 'volume_ma5_ratio',
            'close_ma10_ratio', 'volume_ma10_ratio',
            'close_ma20_ratio', 'volume_ma20_ratio',
            'close_ma60_ratio', 'volume_ma60_ratio',
            'close_ma120_ratio', 'volume_ma120_ratio'
        ]
        training_data = training_data[features_training_data]
    
    
        
        self.data = torch.from_numpy(training_data.values).float()


        
        state = self.data[self.count :self.count+self.view_seq].view(1,-1)
        state = torch.cat([state, torch.Tensor([self.sum_action]).view(1,-1)],dim=1)
        return state
        pass
        

    def step(self,action):
        #continuouse action space
        r_t = d_t= 0
        action = action.item()
        self.sum_action = np.clip(self.sum_action + action, 0,100)
        aa = round(self.sum_action)
        
        delta_a = (self.prev_action - aa)
        
        self.prev_action = aa
        
        
        cost = self.data[self.count,1]*delta_a*0.99
        self.pocket += cost
        
#        r_t = cost
#        r_t = cost
        if self.count+1 == self.count_max:
#            self.pocket += self.data[self.count,1]*(self.prev_action*quantize)
            d_t = 1
            r_t = self.pocket 

        else:
            self.count +=1


        
        state = self.data[self.count :self.count+self.view_seq].view(1,-1)
        state = torch.cat([state, torch.Tensor([self.sum_action]).view(1,-1)],dim=1)
        
        return state ,r_t ,d_t,0
        pass
    
    def vis(self):
        self.win1= vis.line(Y=self.data[:,1],win=self.win1,opts=dict(title='price'))
        self.win2= vis.line(Y=self.data[:,2],win=self.win2,opts=dict(title='vol'))
        





# In[25]:



STEP_MAX =400
WINDOW_SIZE = 2


env = env_stock(STEP_MAX,WINDOW_SIZE,10000000)
env.reset()


#env = NormalizedActions(gym.make("Pendulum-v0"))
ou_noise = OUNoise(1,theta=0.1)

state_dim  = 15*WINDOW_SIZE+1
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


clip = 1
torch.nn.utils.clip_grad_norm_(value_net.parameters(),clip)
torch.nn.utils.clip_grad_norm_(policy_net.parameters(),clip)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


# In[28]:


max_frames  = 12000000
max_steps   = env.count_max
frame_idx   = 0
rewards     = []
batch_size  = 128
#batch_size  = 5


# In[29]:

win_p = vis.line(Y=torch.Tensor([0]),opts=dict(title='policy'))
win_v = vis.line(Y=torch.Tensor([0]), opts=dict(title='value'))
win_r = vis.line(Y=torch.Tensor([0]), opts=dict(title='reward'))
win_a = vis.line(Y=torch.Tensor([0]), opts=dict(title='action'))
win_rew = vis.line(Y=torch.Tensor([0]), opts=dict(title='reward_each_epi'))
win_pocket = vis.line(Y=torch.Tensor([0]))

print(max_steps)



while frame_idx < max_frames:
    state = env.reset()
    
    mem = []
    ou_noise.reset()
    episode_reward = 0
    traj = []
    traj_reward = []
    traj_pocket = []
    
    for step in range(max_steps):
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        
        traj.append(env.sum_action)
        
        next_state, reward, done, _ = env.step(action*1)
        traj_pocket.append(env.pocket.item())
        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) > batch_size and frame_idx%10 == 0:
            ploss,vloss=ddpg_update(batch_size)
            if frame_idx%1000 ==0:
                win_p = vis.line(X=torch.Tensor([frame_idx]), Y=ploss.view(1) , win = win_p, update='append')
                win_v = vis.line(X=torch.Tensor([frame_idx]), Y=vloss.view(1) , win = win_v,  update='append')
            
        state = next_state
        traj_reward.append(reward)
        episode_reward += reward
        frame_idx += 1
        
#        if frame_idx % max(1000, max_steps + 1) == 0:
#            plot(frame_idx, rewards)
            
        if done:
#            replay_buffer.push(mem)
            win_r = vis.line(X=torch.Tensor([frame_idx]), Y=torch.Tensor([episode_reward]).clamp(-1e5,1e5).view(-1) , win = win_r , update='append')
            win_pocket = vis.line(Y=torch.Tensor(traj_pocket).view(-1) , win = win_pocket ,opts=dict(title='pocket_state each step'))
            win_rew = vis.line(Y=torch.Tensor(traj_reward).view(-1) , win = win_rew,opts=dict(title='reward each step'))
            win_a = vis.line(Y=torch.Tensor(traj) , win = win_a,opts=dict(title='sum action each step'))
            env.vis()
            
            break
    
    rewards.append(episode_reward)

