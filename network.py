# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:37:48 2018

@author: JAE
"""


import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import torch.nn.functional as F
from collections import deque
from torch.distributions import Dirichlet, Normal, kl_divergence, Categorical

import random
import gym
class env_torch():
    def __init__(self):
        self.env=gym.make('CartPole-v1')
    def reset(self):
        return torch.from_numpy(self.env.reset()).type(torch.float32)
    def step(self,action):
        s_t,r_t,done, _ = self.env.step(action)
        s_t = torch.from_numpy(s_t).type(torch.float32)
        return s_t,r_t,not done,_
    def close(self):
        self.env.close()



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class baseclass(nn.Module):
    def __init__(self):
        super(baseclass,self).__init__()
        pass
    def print_weights(self):
        for net in self.net:
            print(net.weight.data)
    def print_bias(self):
        for net in self.net:
            print(net.bias.data)
            
class Actor(baseclass):
    def __init__(self,netsize):
        super(Actor,self).__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(netsize[0], netsize[1]))
        self.net.append(nn.Linear(netsize[1], netsize[2]))
        self.net.append(nn.Linear(netsize[2], netsize[3]))
        
        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm1d(netsize[1]))
        self.bn.append(nn.BatchNorm1d(netsize[2]))
        
        self.init_weights()
        
    def init_weights(self, init_w=3e-3):
        self.net[0].weight.data = fanin_init(self.net[0].weight.data.size())
        self.net[1].weight.data = fanin_init(self.net[1].weight.data.size())
        self.net[2].weight.data.uniform_(-init_w, init_w)
        
        # add bn initialize context


    
    def forward(self, x, EN_Batchnorm = False):
        x = self.net[0](x)
        x = self.bn[0](x) if EN_Batchnorm else x
        x = F.leaky_relu(x)

        x = self.net[1](x)
        x = self.bn[1](x) if EN_Batchnorm else x
        x = F.leaky_relu(x)

        x = self.net[2](x)
        x = F.tanh(x)
        return x
        
class Critic(baseclass):
    def __init__(self, netsize,nb_action):
        super(Critic, self).__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(netsize[0], netsize[1]))
        self.net.append(nn.Linear(netsize[1]+nb_action, netsize[2]))
        self.net.append(nn.Linear(netsize[2], netsize[3]))
        
        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm1d(netsize[1]))
        self.bn.append(nn.BatchNorm1d(netsize[2]))
        
        self.init_weights()
        
    def init_weights(self, init_w=3e-3):
        self.net[0].weight.data = fanin_init(self.net[0].weight.data.size())
        self.net[1].weight.data = fanin_init(self.net[1].weight.data.size())
        self.net[2].weight.data.uniform_(-init_w, init_w)
    
        # add bn initialize context
    
    def forward(self, x, action, EN_Batchnorm = False):
        x = self.net[0](x)
        x = self.bn[0](x) if EN_Batchnorm else x
        x = F.leaky_relu(x)
        action = action.type(torch.float32)
        out = torch.cat([x,action],1)
        out = self.net[1](out)
        out = self.bn[1](out) if EN_Batchnorm else x 
        out = F.leaky_relu(out)

        out = self.net[2](out)
        return out

    

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class Agent():
    def __init__(self,nb_states,nb_action, mem_size):
        self.nb_states = nb_states
        self.nb_action = nb_action
        self.lr = 0.001
        
        self.batch_size =32
        self.seq_size = 1

        self.tau = 0.3
        self.discount = 0.99

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action

        
        self.actor = Actor([nb_states,32,32,nb_action])
        self.actor_target = Actor([nb_states,32,32,nb_action])
        self.actor_optim = Adam(self.actor.parameters(),lr=self.lr)
        
        self.critic = Critic([nb_states,32,32,1],nb_action)
        self.critic_target = Critic([nb_states,32,32,1],nb_action)
        self.critic_optim = Adam(self.critic.parameters(),lr=self.lr)
        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        self.epi_memory = deque(maxlen=mem_size)
        self.random = Dirichlet(torch.ones([1,nb_action]))
        
        
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def get_q_value(self,s_t ,a_t):
        q = self.critic(s_t,a_t)
        return q
    
    
    def get_action(self, s_t, random = False, eps = None ):
        if random:
            action = self.random.rsample()
            self.a_t = action
            return action
        
        out = self.actor(s_t)
        epsilon = self.epsilon if eps is None else eps 
        action = (1-epsilon)*out + (epsilon)*self.random.rsample()
#        action = action.max(1)[1].unsqueeze(1)
               
        self.a_t = action
        return action
    
    
    def mem_append(self,episodic_info):
        self.epi_memory.append(episodic_info)
        pass
    
    def mem_sample(self,batch_size,seq_size):
        
        batch_data = random.sample(self.epi_memory,batch_size)
        
        batch_seq = []
        for data in batch_data:
            subseq = []
            start = random.randint(0,len(data)-seq_size)
            for se in range(seq_size):
                subseq.append(data[start+se])
            batch_seq.append(subseq)
        
        return batch_seq
    
    
    def load_model(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    
    def set_epsilon(self,val):
        self.epsilon = val
    
    def update_policy(self):
        batch_size =4
        seq_size =2
        b_st,b_at,b_rt,b_st_1,b_done = [],[],[],[],[]
        b_info = self.mem_sample(batch_size,seq_size)

        for info in b_info:
            for seq in info:
                b_st.append(seq[0])
                b_at.append(seq[1])
                b_rt.append(seq[2])
                b_st_1.append(seq[3])
                b_done.append(seq[4])

        batch_shape = (batch_size,-1)
        #batch_shape = (batch_size,seq_size,-1)


        b_st = torch.stack(b_st).reshape(batch_shape)
        b_at = torch.Tensor(b_at).reshape(batch_shape)
        b_rt = torch.Tensor(b_rt).reshape(batch_shape)
        b_st_1 = torch.stack(b_st_1).reshape(batch_shape)
        b_done = torch.Tensor(b_done).reshape(batch_shape)
       with torch.no_grad():
           next_q = self.critic_target(b_st,self.actor_target(b_st_1))
           target_q_batch = b_rt + self.discount * b_done * next_q

        self.critic.zero_grad()
        q_batch = self.critic_target(b_st,b_at)
        value_loss = F.mse_loss(q_batch,target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = -self.critic(b_st,self.actor(b_st)).mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target,self.actor,self.tau)
        soft_update(self.critic_target,self.critic,self.tau)
        return value_loss , policy_loss

        
#        next_q_values = self.critic_target([
#            to_tensor(next_state_batch, volatile=True),
#            self.actor_target(to_tensor(next_state_batch, volatile=True)),
#        ])
#        next_q_values.volatile=False
#
#        target_q_batch = to_tensor(reward_batch) + \
#            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values
#        target_q_batch = target_q_batch.detach()
#        # Critic update
#        self.critic.zero_grad()
#
#        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
#        
#        value_loss = criterion(q_batch, target_q_batch)
#        value_loss.backward()
#        self.critic_optim.step()
#
#        # Actor update
#        self.actor.zero_grad()
#
#        policy_loss = -self.critic([
#            to_tensor(state_batch),
#            self.actor(to_tensor(state_batch))
#        ])
#
#        policy_loss = policy_loss.mean()
#        policy_loss.backward()
#        self.actor_optim.step()
#
#        # Target update
#        soft_update(self.actor_target, self.actor, self.tau)
#        soft_update(self.critic_target, self.critic, self.tau)

    
    
def _test_():
    #actor critic 테스트
    nb_states = 5
    nb_action = 3
    xx = torch.rand([1,nb_states])
    action = torch.rand([1,nb_action])
    
    ac = Actor([nb_states,3,3,3])
    print(ac(xx))
    #ac.print_weights()
    #ac.print_bias()
    
    cr = Critic([nb_states,3,3,1],nb_action)
    print(cr(xx,action))
    
def _test2_():
    #랜덤 action 테스트
    ag = Agent(nb_states=5,nb_action=5)
    state = torch.rand([3,5])
    ag.get_q_value(state,ag.get_action(state))
    
#    arr = [ag.get_action(state) for x in range(1000)]
#    nparr = np.array(arr)
#    print(np.unique(nparr,return_counts=True))
    


_test2_()


agent = Agent(nb_states=4,nb_action=2,mem_size=10000)

env = env_torch()
eps = 1.0

for episode in range(1000):
    mem = []
    s_t = env.reset()
    for T in range(201):
        a_t = agent.get_action(s_t,eps=eps)
        cate = Categorical(a_t)
        acc = cate.sample().item()
        s_t_1, r_t,done,_=env.step(acc)
        mem.append([s_t,a_t,r_t,s_t_1,done])
        s_t = s_t_1

        if T%10 ==0 and episode>buf_fill and eps>0.001:
            v_loss, p_loss = agent.update_policy()
            eps = eps-0.00001 if eps>0 else 0 
            print(v_loss, p_loss, eps)

        if not done or T>=200:
            print('episode ',episode, 'max T',T)
            agent.mem_append(mem)
            break

env.close()

