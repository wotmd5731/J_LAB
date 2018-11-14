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
import sys

hidden_dim = 128
batch_size = 8
burn_in_length = 10
sequences_length = 20
feature_state = (3,64,64)
feature_reward = 1
feature_action = 4



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
        self.size = (((state_shape[1]-1)//4)-3)//2-1
        
        self.lstm = torch.nn.LSTMCell(input_size=64*self.size*self.size , hidden_size=hidden_dim)
#        input of shape (batch, input_size): tensor containing input features
#        hidden of shape (batch, hidden_size)
        
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( hidden_dim, hidden_dim),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(hidden_dim, 1)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.front(x)
        x = x.view(-1, 64 * self.size * self.size)
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
    

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def actor_process(rank, shared_state, shared_queue, max_frame = 1 ):
    print('{} actor process start '.format(rank))
    
    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    n_step = 5
    

    env = gym.make("Breakout-v0")
    policy_epsilon = 0.3*rank
    action = 0
    gamma_t = 0.997
    frame = 0
    total_reward = []
    ot= obs_preproc(env.reset())
    hx = torch.zeros([1,hidden_dim])
    cx = torch.zeros([1,hidden_dim])
    while frame < max_frame:
        copy_hx, copy_cx = hx.clone(), cx.clone()
        
        local_buf = []
        for seq in range(sequences_length):
            frame+=1
            with torch.no_grad():
                Qt, (hx,cx) = Q_main(ot,(hx,cx))
                #e greedy
                if random.random() >= policy_epsilon:
                    action =  torch.argmax(Qt,dim=1).item()
                else:
                    action = random.randint(0,feature_action-1)
            
            ot_1, rt,dt,_  = env.step(action)
            env.render()
            
            total_reward.append(rt)
            ot_1 = obs_preproc(ot_1)
            local_buf.append([ot,action,rt,gamma_t])
            
            ot = ot_1
            if dt == True:
                ot= obs_preproc(env.reset())
                print('#{} total reward: {}'.format(rank,sum(total_reward)))
                total_reward = []
                break
            if frame % 100 == 0:
                Q_main.load_state_dict(shared_state["Q_state"])
#                hard_update(Q_main,shared_state["Q_state"])
        if seq>= burn_in_length + n_step :
            for s in range(seq+1,sequences_length):
                local_buf.append([ot,action,rt,0])
        state, action, reward,gamma = map(np.stack,zip(*local_buf))
        local_buf=[]
        shared_queue.put([state,reward,gamma,copy_hx,copy_cx])
            
        
        
#        global_buf.append(Global_Transition(seq,local_buf,copy_hidden,dt))
        
                
    print('{} actor process done '.format(rank))
#        if dt == True:
#            break
        
def h_func(x):
    epsilon= 10e-2
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
def h_inv_func(x):
    epsilon= 10e-2
    return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
    
def learner_process(rank , shared_state, shared_queue, max_frame =1 ):
    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)
    Q_target = Duelling_LSTM_DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    Q_target.load_state_dict(shared_state["Q_state"])
    
    
    value_optimizer  = optim.Adam(Q_main.parameters(),  lr=0.0001)
    global_buf = deque(maxlen = 10000)

    n_step = 5
    gamma = 0.997
    frame = 0
    global_buf = []
    
    i=0
    while shared_queue.qsize() <= batch_size:
        i+=1
    
    
    while frame<max_frame:
        for i in range(shared_queue.qsize()):
            global_buf.append(shared_queue.get())
        frame+=1

        batch = random.sample(global_buf, batch_size)
        state, action, reward, gamma, copy_hx, copy_cx = map(np.stack, zip(*batch))
        st = torch.from_numpy(state).view(sequences_length,batch_size,feature_state[0],feature_state[1],feature_state[2]).float()
        at = torch.from_numpy(action).view(sequences_length,batch_size).long()
        rt = torch.from_numpy(reward).view(sequences_length,batch_size).float()
        gamt = torch.from_numpy(gamma).view(sequences_length,batch_size).float()
        
        hx_m = torch.from_numpy(copy_hx).view(batch_size,hidden_dim)
        cx_m = torch.from_numpy(copy_cx).view(batch_size,hidden_dim)
        
        hx_t = hx_m.clone()
        cx_t = cx_m.clone()
        
        with torch.no_grad():
            for i in range(burn_in_length):
                _, (hx_m, cx_m) = Q_main(st[i], (hx_m, cx_m))
                _, (hx_t, cx_t) = Q_target(st[i], (hx_t, cx_t))
                
        hx_double_m = hx_m.clone()
        cx_double_m = cx_m.clone()
        hx_double_t = hx_t.clone()
        cx_double_t = cx_t.clone()
        
        Q_tilda = []
        for i in range(burn_in_length, sequences_length):
            target_Q_v, (hx_m, cx_m) = Q_target(st[i], (hx_m, cx_m))
            a_star = torch.argmax(target_Q_v, dim =1)
            
            main_Q_v , (hx_t, cx_t) = Q_main(st[i], (hx_t, cx_t))
            
            Q_tilda.append(main_Q_v.gather(1,a_star.view(batch_size,-1)))
            
        loss = 0
        for i in range(burn_in_length, sequences_length-n_step):
            sub_ten = torch.stack([rt[i+k]*(gamt[i+k]**k) for k in range(n_step)],dim=1)
            rt_sum = torch.sum(sub_ten, dim=1)
            inv_scaling_Q = h_inv_func(Q_tilda[i - burn_in_length + n_step].view(-1))
            y_t_hat = h_func(rt_sum + gamt[i+n_step]**n_step * inv_scaling_Q)
            
            main_Q_value , (hx_m, cx_m ) = Q_main(st[i], (hx_m, cx_m))
            Q_value = main_Q_value.gather(1,at[i].view(batch_size,-1)).view(-1)
            
            loss += F.mse_loss(Q_value, y_t_hat)
            
        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()
        print('loss:',loss.item())
        
        
        tau = 0.3
        for target_param, param in zip(Q_target.parameters(),Q_main.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau )
        shared_state["Q_state"] = Q_main.state_dict()

        
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
        if frame % 3 == 0:
            tau = 0.3
            for target_param, param in zip(Q_target.parameters(),Q_main.parameters()):
                target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau )
            shared_state["Q_state"] = Q_main.state_dict()
#for i in range(1000):
#    actor_process(global_buf)
#    learner_process(global_buf)
    


#
#mp_manager = mp.Manager()
#shared_state = mp_manager.dict()
#shared_mem = mp_manager.Queue()
#
#
#ReplayManager = BaseManager()
#ReplayManager.start()
#replay_mem = ReplayManager.Memory(replay_params["soft_capacity"],  replay_params)
#
#    

if __name__ == '__main__':
    
    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)

    num_processes = 1
    
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_queue = manager.Queue()
    shared_state["Q_state"] = Q_main.state_dict()
    actor_process(0, shared_state, shared_queue,1000)
    learner_process(0, shared_state, shared_queue,10)
    
#    learner_procs = mp.Process(target=actor_process, args=(i, shared_state, shared_queue,))
#    learner_procs.start()
#    
#    actor_procs = []
#    for i in range(num_processes):
#        print(i)
#        actor_proc = mp.Process(target=actor_process, args=(i, shared_state, shared_queue,))
#        actor_proc.start()
#        actor_procs.append(actor_proc)
#    for act in actor_procs:
#        act.join()    
#    learner_procs.join()
    






