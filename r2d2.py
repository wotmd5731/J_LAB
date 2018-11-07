# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import random
import numpy as np
from collections import namedtuple
from duelling_network import DuellingDQN
from env import make_local_env
import gym


env_conf = {"state_shape": (3, 84, 84),
            "action_dim": 4,
            "name": "Breakout-v0"}

class Duelling_LSTM_DQN(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Duelling_LSTM_DQN, self).__init__()
        self.input_shape = state_shape
        self.action_dim = action_dim
        self.front = torch.nn.Sequential(torch.nn.Conv2d(state_shape[0], 64, 8, stride=4),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(64, 64, 4, stride=2),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(64, 64, 3, stride=1),
                                          torch.nn.ReLU())

        self.lstm = torch.nn.LSTMCell(input_size=64*7*7 , hidden_size=512)
#        input of shape (batch, input_size): tensor containing input features
#        hidden of shape (batch, hidden_size)
        
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( 512, 512),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(512, 512),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(512, 1)
        self.advantage = torch.nn.Linear(512, action_dim)

    def forward(self, x, hidden):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.front(x)
        x = x.view(-1, 64 * 7 * 7)
        x ,hidden = self.lstm(x, hidden)
#        output of shape (seq_len, batch, num_directions * hidden_size)
        
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return action_value, hidden

env = gym.make(env_conf['name'])

#Q = Duelling_LSTM_DQN(env_conf['state_shape'], env_conf['action_dim'])




def update():
    batch_size = 8
    burn_in_length = 40
    sequences_length = 80
    feature_state = (3,84,84)
    feature_reward = 1
    feature_action = 4
    
    
    
    ot_burn = torch.rand([batch_size,burn_in_length , feature_state])
    rt_burn = torch.rand([batch_size,burn_in_length , feature_reward])
    at_burn = torch.rand([batch_size,burn_in_length , feature_action])
    
    ot = torch.rand([batch_size, sequences_length, feature_state])
    rt = torch.rand([batch_size, sequences_length, feature_reward])
    at = torch.rand([batch_size, sequences_length, feature_action])
    
#        input of shape (batch, input_size): tensor containing input features
#        hidden of shape (batch, hidden_size)
    hidden_target = (torch.zeros([batch_size, 512]),torch.zeros([batch_size, 512]))
    hidden_main = (torch.zeros([batch_size, 512]),torch.zeros([batch_size, 512]))
    hidden_main_copy = hidden_main
    hidden_target_copy = hidden_target
    # get replay buffer
    
    for i in range(burn_in_length):
        _ , hidden_main = Q_main(ot_burn[i],hidden_main)
        _ , hidden_target = Q_target(ot_burn[i],hidden_target)
    
    n_step = 5 # n-step 
    t = 0 # init time step
    gamma = 0.997
    def h_func(x):
        epsilon= 10e-2
        return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
    def h_inv_func(x):
        epsilon= 10e-2
        return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
        


    #calc Q tilda 
    Q_tilda=[0 for i in range(sequences_length)]
    for i in range(sequences_length):
        
        target_Q_value, hidden_target = Q_target(ot[i],hidden_target)
        a_star = torch.argmax(target_Q_value)
        
        main_Q_value, hidden_main = Q_main(ot[i],hidden_main)
        Q_tilda[i] = main_Q_value[a_star]

    
    hidden_main = hidden_main_copy
    hidden_target = hidden_target_copy

    for i in range(sequences_length):
        
        rt_sum = torch.sum( [rt[i+k]*gamma**k for k in range(n_step)] )
        inv_scaling_Q = h_inv_func( Q_tilda[i+n_step] )
        y_t_hat = h_func(rt_sum + gamma**n_step * inv_scaling_Q)
        
        main_Q_value, hidden_main = Q_main(ot[i] , hidden_main)
        
        loss = 1/2*(y_t_hat - main_Q_value)**2
        loss.backward()
        
        
        
        
        
        
        


Q_main = Duelling_LSTM_DQN(env_conf['state_shape'], env_conf['action_dim'])
Q_target = Duelling_LSTM_DQN(env_conf['state_shape'], env_conf['action_dim'])

    









