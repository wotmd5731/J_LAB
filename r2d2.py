# -*- coding: utf-8 -*-
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

env_conf = {"state_shape": (3, 84, 84),
            "action_dim": 4,
            "name": "Breakout-v0"}
batch_size = 8
burn_in_length = 10
sequences_length = 20
feature_state = (3,84,84)
feature_reward = 1
feature_action = 4
env = gym.make(env_conf['name'])


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
        hidden = self.lstm(x, hidden)
        x=hidden[0]
        
#        output of shape (seq_len, batch, num_directions * hidden_size)
        
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return action_value, hidden


#Q = Duelling_LSTM_DQN(env_conf['state_shape'], env_conf['action_dim'])

def obs_preproc(x):
    return torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)/256
    
Transition = namedtuple('Transition', ['S', 'A', 'R', 'Gamma'])
Global_Transition = namedtuple('Global_Transition', ['seq', 'local_buf', 'hidden', 'done'])

global_buf = deque(maxlen = 1000)

Q_main = Duelling_LSTM_DQN(feature_state, feature_action)
Q_target = Duelling_LSTM_DQN(feature_state, feature_action)

value_optimizer  = optim.Adam(Q_main.parameters(),  lr=0.0001)


def actor_process(global_buf):
    max_frame = 2000
    policy_epsilon = 0.1
    action = 0
    gamma_t = 0.997
    frame = 0
    total_reward = []
    ot= obs_preproc(env.reset())
    hidden_main = (torch.zeros([1,512]), torch.zeros([1,512]))
    while frame < max_frame:
        local_buf = []
        for seq in range(sequences_length):
            if seq == sequences_length//2:
                copy_hidden = copy.deepcopy(hidden_main)
            with torch.no_grad():
                Qt, hidden_main = Q_main(ot,hidden_main)
                #e greedy
                if random.random() >= policy_epsilon:
                    action =  torch.argmax(Qt,dim=1).item()
                else:
                    action = random.randint(0,feature_action-1)
            
            ot_1, rt,dt,_  = env.step(action)
            env.render()
            
            total_reward.append(rt)
            ot_1 = obs_preproc(ot_1)
            local_buf.append(Transition(ot,action,rt,gamma_t))
            
            ot = ot_1
            if dt == True:
                ot= obs_preproc(env.reset())
                print('total reward: {}'.format(sum(total_reward)))
                break
        seq+=1
        global_buf.append(Global_Transition(seq,local_buf,copy_hidden,dt))
        frame+=seq
        if dt == True:
            break
        
def h_func(x):
    epsilon= 10e-2
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
def h_inv_func(x):
    epsilon= 10e-2
    return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
    
def learner_process(global_buf):
    n_step = 5
    gamma = 0.997
    frame = 0
    max_frame = 1
    
    
    while frame<max_frame:
        frame+=1
        T_loss = 0
        
        for bat in range(batch_size):
            
            while True:
                idx = random.randint(0,len(global_buf)-1-1)
                if global_buf[idx].done == False:
                    break
            
            
            with torch.no_grad():
                #burn in
                burn_list = global_buf[idx].local_buf
                burn_seq = global_buf[idx].seq
                
                hidden_main = (global_buf[idx].hidden[0].clone(),global_buf[idx].hidden[1].clone())
                hidden_target = (global_buf[idx].hidden[0].clone(),global_buf[idx].hidden[1].clone())
                
                
                ot_burn = [ burn_list[sequences_length - burn_in_length +k].S for k in range(burn_in_length)]
            
                for i in range(burn_in_length):
                    _, hidden_main = Q_main(ot_burn[i], hidden_main)
                    _, hidden_target = Q_main(ot_burn[i], hidden_target)
                
                
                stored_hidden_main = (hidden_main[0].clone(),hidden_main[1].clone())
                stored_hidden_target = (hidden_target[0].clone(),hidden_target[1].clone())
    
                
                #train
                train_list = global_buf[idx+1].local_buf
                train_seq = global_buf[idx+1].seq
                
                St = [train_list[i].S for i in range(train_seq)]
                At = torch.Tensor([train_list[i].A for i in range(train_seq)]).long()
                Rt = torch.Tensor([train_list[i].R for i in range(train_seq)])
                Gamma_t = torch.Tensor([train_list[i].Gamma for i in range(train_seq)])
                
                
                
                
                #calc Q tilda
                Q_tilda = []
                for i in range(train_seq):
                    target_Q_v, hidden_target = Q_target(St[i], hidden_target)
                    a_star = torch.argmax(target_Q_v, dim =1)
                    
                    main_Q_v , hidden_main = Q_main(St[i], hidden_main)
                    Q_tilda.append(torch.index_select(main_Q_v ,1,a_star))
    
                
            
            
            
            hidden_main = (stored_hidden_main[0].clone(),stored_hidden_main[1].clone())
            hidden_target = (stored_hidden_target[0].clone(),stored_hidden_target[1].clone())
            
            
            
            for i in range(train_seq-n_step):
                rt_sum = torch.sum( torch.stack([Rt[i+k]*gamma**k for k in range(n_step)]) )
                inv_scaling_Q = h_inv_func( Q_tilda[i+n_step] )
                y_t_hat = h_func(rt_sum + gamma**n_step * inv_scaling_Q)
                
                main_Q_value, hidden_main = Q_main(St[i] , hidden_main)
                
                loss = 1/2*(y_t_hat - torch.index_select(main_Q_value, 1, At[i] ) )**2
                
                T_loss += loss
                
                
        value_optimizer.zero_grad()        
        (T_loss/batch_size).backward()
        value_optimizer.step()
            
        print('batch:{} T_loss:{} '.format(bat,T_loss.item()))
    
        soft_tau = 0.3
        for target_param, param in zip(Q_target.parameters(), Q_main.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
        
        
for i in range(1000):
    actor_process(global_buf)
#    learner_process(global_buf)
    









