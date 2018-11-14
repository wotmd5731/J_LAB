# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 01:45:23 2018

@author: JAE
"""

import torch
import torch.multiprocessing as mp

import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import gym
import copy
import sys


hidden_dim = 128
batch_size = 8
burn_in_length = 10
sequences_length = 20
#feature_state = (1,84,84)
feature_state = (4)
feature_reward = 1
feature_action = 2
n_step = 5
gamma_t = 0.997
IMG_GET_RENDER = False   
#IMG_GET_RENDER = True   



def obs_preproc(x):
    if IMG_GET_RENDER == False   :
        x = torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)
        return x
    x= np.dot(x, np.array([[0.299, 0.587, 0.114]]).T)
    x= np.reshape(x, (1, x.shape[1], x.shape[0]))
    x = torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)/255
    return x



class DQN(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DQN, self).__init__()
        self.input_shape = state_shape
        self.action_dim = action_dim
#        self.front = torch.nn.Sequential(torch.nn.Conv2d(state_shape[0], 64, 5, stride=3),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(64, 64, 3, stride=3),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(64, 64, 3, stride=1),
#                                          torch.nn.ReLU())
        self.size = 2
        self.front = torch.nn.Sequential(torch.nn.Linear( 4, 64*self.size*self.size),
                                                      torch.nn.ReLU())
        
        
#        self.lstm = torch.nn.LSTMCell(input_size=64*self.size*self.size , hidden_size=hidden_dim)
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( 64*self.size*self.size, hidden_dim),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(64*self.size*self.size, hidden_dim),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(hidden_dim, 1)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.front(x)
        x = x.view(-1, 64 * self.size * self.size)
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return action_value
    





def actor_process(rank, shared_state, shared_queue, max_frame = 1 ):
    print('{} actor process start '.format(rank))
    
    Q_main = DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    

     

#    env = gym.make("Breakout-v0")
    env = gym.make('CartPole-v0')
    policy_epsilon = 0.2*rank+0.05
    action = 0
    frame = 0
    total_reward = []
    
    obs =env.reset()
    if IMG_GET_RENDER:
        obs =env.render(mode='rgb_array')
    ot= obs_preproc(obs)
    
    
    while frame < max_frame:
        
        for seq in range(sequences_length):
            frame+=1
            with torch.no_grad():
                Qt = Q_main(ot)
                #e greedy
                if random.random() >= policy_epsilon:
                    action =  torch.argmax(Qt,dim=1).item()
                else:
                    action = random.randint(0,feature_action-1)
            
            ot_1, rt,dt,_  = env.step(action)
            
            if IMG_GET_RENDER:
                obs =env.render(mode='rgb_array')
            ot_1 = obs_preproc(obs)
            gamma_t = 0 if dt else 0.99
            
            shared_queue.put([ot,action,rt,gamma_t,ot_1])
            total_reward.append(rt)
            
            ot = ot_1
            
            if dt == True:
                obs =env.reset()
                if IMG_GET_RENDER:
                    obs =env.render(mode='rgb_array')
                ot = obs_preproc(obs)
                if rank == 0:
                    print('#{} total reward: {}'.format(rank,sum(total_reward)))
                total_reward = []
                break
            if frame % 100 == 0:
                Q_main.load_state_dict(shared_state["Q_state"])
                
    print('{} actor process done '.format(rank))
#    state, action, reward,gamma ,next_state= map(np.stack,zip(*local_buf))
    
    

def learner_process(rank , shared_state, shared_queue, max_frame =1 ):
    Q_main = DQN(feature_state, feature_action)
    Q_target = DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    Q_target.load_state_dict(shared_state["Q_state"])
    
    
    value_optimizer  = optim.Adam(Q_main.parameters(),  lr=0.00001)
    global_buf = deque(maxlen = 10000)
    frame = 0
    i=0
    while len(global_buf) <= 100:
        global_buf.append(shared_queue.get())
    
    
    
    while frame<max_frame:
        for i in range(shared_queue.qsize()):
            global_buf.append(shared_queue.get())
        frame+=1

        batch = random.sample(global_buf, batch_size)
        state, action, reward, gamma, next_state = map(np.stack, zip(*batch))
        st = torch.from_numpy(state).view(batch_size,feature_state[0],feature_state[1],feature_state[2]).float()
        at = torch.from_numpy(action).view(batch_size).long()
        rt = torch.from_numpy(reward).view(batch_size).float()
        gamt = torch.from_numpy(gamma).view(batch_size).float()
        st_1 = torch.from_numpy(next_state).view(batch_size,feature_state[0],feature_state[1],feature_state[2]).float()
        
        with torch.no_grad():
            exQ = rt + Q_main(st_1).gather(1,torch.argmax(Q_target(st_1),dim=1).view(batch_size,-1)).view(-1) * gamt
        
        
        Qv = Q_main(st).gather(1,at.view(batch_size,-1)).view(-1)
        
        value_optimizer.zero_grad()
        loss = F.mse_loss (Qv, exQ)
#        print(loss.item())
        loss.backward()
        value_optimizer.step()
        
        if frame%100 == 0:
            Q_target.load_state_dict(Q_main.state_dict())
            shared_state["Q_state"] = Q_main.state_dict()

        
        
        
        
if __name__ == '__main__':
    
    Q_main = DQN(feature_state, feature_action)

    num_processes = 4
    
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_queue = manager.Queue()
    shared_state["Q_state"] = Q_main.state_dict()
    
    actor_process(0, shared_state, shared_queue,100)
    learner_process(0, shared_state, shared_queue,2)
    
#    learner_procs = mp.Process(target=learner_process, args=(999, shared_state, shared_queue,10000))
#    learner_procs.start()
#    
#    actor_procs = []
#    for i in range(num_processes):
#        print(i)
#        actor_proc = mp.Process(target=actor_process, args=(i, shared_state, shared_queue,10000))
#        actor_proc.start()
#        actor_procs.append(actor_proc)
#    for act in actor_procs:
#        act.join()    
#    learner_procs.join()
