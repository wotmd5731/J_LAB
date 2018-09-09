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
from torch.distributions import Dirichlet
import random


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
    
    
    def forward(self, x):
        x = self.net[0](x)
        x = F.leaky_relu(x)
        x = self.net[1](x)
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
        self.bn.append(nn.BatchNorm1d(netsize[1]+nb_action))
        self.bn.append(nn.BatchNorm1d(netsize[2]))
        
        self.init_weights()
        
    def init_weights(self, init_w=3e-3):
        self.net[0].weight.data = fanin_init(self.net[0].weight.data.size())
        self.net[1].weight.data = fanin_init(self.net[1].weight.data.size())
        self.net[2].weight.data.uniform_(-init_w, init_w)
    
    
    def forward(self, x, action):
        x = self.net[0](x)
        x = F.leaky_relu(x)
        out = torch.cat([x,action],1)
        out = self.net[1](out)
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
    def __init__(self,nb_states,nb_action):
        self.nb_states = nb_states
        self.nb_action = nb_action
        self.lr = 0.001
        
        self.batch_size = 2
        self.seq_size = 1
        self.tau = 0.1
        self.discount = 0.99

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action

        
        self.actor = Actor([nb_states,3,3,nb_action])
        self.actor_target = Actor([nb_states,3,3,nb_action])
        self.actor_optim = Adam(self.actor.parameters(),lr=self.lr)
        
        self.critic = Critic([nb_states,3,3,1],nb_action)
        self.critic_target = Critic([nb_states,3,3,1],nb_action)
        self.critic_optim = Adam(self.critic.parameters(),lr=self.lr)
        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        self.epi_memory = deque(maxlen=10)
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


    
    def get_action(self, s_t, random = False ):
        if random:
            action = self.random.rsample().max(1)[1].item()
            self.a_t = action
            return action
        
        out = self.actor(self.preprocess(s_t))
        tau = 0.8
        action = (1-tau)*out + (tau)*self.random.rsample()
        action = action.max(1)[1].item()
               
        self.a_t = action
        return action
    
    
    def mem_append(self,episodic_info):
        self.epi_memory.append(episodic_info)
        pass
    
    def mem_sample(self):
        
        batch_size = self.batch_size
        batch_data = random.sample(self.epi_memory,batch_size)
        seq_size = self.seq_size
        
        batch_seq = []
        for data in batch_data:
            subseq = []
            start = random.randint(0,len(data)-seq_size-1)
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
    
    
    
    def update_policy(self):
        # Sample batch
        batch_info = np.array(self.mem_sample())
        # state pre process
        batch_s_t = batch_info[:,:,0]
        
        
        batch_a_t = batch_info[:,:,1]
        batch_r_t = batch_info[:,:,2]
        batch_s_t_1 = batch_info[:,:,3]
        batch_done = batch_info[:,:,4]
        
        
        # Prepare for the target q batch
        next_q = self.critic_target(batch_s_t_1, self.actor_target(batch_s_t_1) )
        
        
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
    state = torch.rand([1,5])
    arr = [ag.get_action(state) for x in range(1000)]
    nparr = np.array(arr)
    print(np.unique(nparr,return_counts=True))





agent = Agent(nb_states=4,nb_action=2)

import gym
env = gym.make('CartPole-v1')
global_count = 0
episode = 0
while episode < 7:
    episode += 1
    T=0
    mem = []
    s_t = env.reset()
#    args.epsilon -= 0.8/args.max_episode_length
    while T < 100:
        T += 1
        a_t = agent.get_action(s_t,random=True)
        s_t_1 , r_t , done, _ = env.step(a_t)
#        env.render()
        mem.append([s_t, a_t, r_t, s_t_1, done])
        
        s_t = s_t_1
#        agent.update_policy()
#        agent.target_update()
        
#        if global_count % args.replay_interval == 0 :
#            agent.basic_learn(memory)
#        if global_count % args.target_update_interval == 0 :
#            agent.target_dqn_update()

        if done or T>10 :
            agent.mem_append(mem)
            break
env.close()

agent.update_policy()