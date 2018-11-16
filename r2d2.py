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
import time
import visdom

vis = visdom.Visdom()
vis.close()


USE_MP = False

batch_size = 32
burn_in_length = 4
sequences_length = 12

hidden_dim = 128
n_step = 4
def_gamma = 0.997
def_global_buf_maxlen = 10000
def_lr = 0.0001
def_soft_update_tau = 0.3
def_learner_update_step = 50
def_actor_update_step = 500
train_start_size = 500
num_processes = 8
learner_frame_interval = 0.01 #sec

CONF = 'cartpole_state_4'

if CONF == 'cartpole_state_4':
    GAME_NAME = 'CartPole-v1'
    feature_reward = 1
    feature_action = 2
    feature_state = (1,4)
    FRONT_CNN = False
    IMG_GET_RENDER = False
elif CONF == 'cartpole_state_img':
    GAME_NAME = 'CartPole-v1'
    feature_reward = 1
    feature_action = 2
    feature_state = (1,64,64)
    FRONT_CNN = True
    IMG_GET_RENDER = True
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def append(self, info):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = info
        self.position = (self.position +1)%self.capacity

    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)

    def __len__(self):
        return len(self.buffer)



class Duelling_LSTM_DQN(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Duelling_LSTM_DQN, self).__init__()
        self.input_shape = state_shape
        self.action_dim = action_dim
        if FRONT_CNN:
            self.front = torch.nn.Sequential(torch.nn.Conv2d(state_shape[0], 64, 8, stride=4),
                                              torch.nn.ReLU(),
                                              torch.nn.Conv2d(64, 64, 4, stride=2),
                                              torch.nn.ReLU(),
                                              torch.nn.Conv2d(64, 64, 3, stride=1),
                                              torch.nn.ReLU())
            self.size = (((state_shape[1]-1)//4)-3)//2-1
            self.lstm_in_size = 64*self.size**2
        else:
            self.front = torch.nn.Sequential(torch.nn.Linear(feature_state[1] , hidden_dim),
                                                      torch.nn.ReLU())
            self.lstm_in_size = hidden_dim
        
        self.lstm = torch.nn.LSTMCell(input_size=self.lstm_in_size , hidden_size=hidden_dim)
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
        x = x.view(-1,self.lstm_in_size)
        hidden = self.lstm(x, hidden)
        x=hidden[0]
        
#        output of shape (seq_len, batch, num_directions * hidden_size)
        
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return action_value, hidden


#Q = Duelling_LSTM_DQN(env_conf['state_shape'], env_conf['action_dim'])

def obs_preproc(x):
    if IMG_GET_RENDER ==False:  
        return torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)
    x = np.dot(x, np.array([[0.299, 0.587, 0.114]]).T)
    x = np.reshape(x, (1,x.shape[1], x.shape[0]))
    return torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)/255

    

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def actor_process(rank, shared_state, shared_queue, max_frame , dev, num_processes):
    
    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    
    env = gym.make(GAME_NAME)
    policy_epsilon = 0.05**((rank+1)/num_processes)
    win_r = vis.line(Y=torch.Tensor([0]), opts=dict(title ='reward'+str(policy_epsilon)))
    print(f'#{rank} actor process start p:{policy_epsilon} ')

    action = 0
    gamma_t = def_gamma
    frame = 0
    ttime = time.time()

    total_reward = []
    prev_list = []
    local_buf = []
    dt = True

    while frame < max_frame:
        for seq in range(sequences_length//2):
            if dt==True :
                win_r = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([sum(total_reward)]), win= win_r , update ='append')
                print(f'#{rank} frame:{frame:5d} total_reward: {sum(total_reward)}, step:{len(total_reward)}, time:{time.time()-ttime}')
                ttime = time.time()
                #env reset
                obs = env.reset()
                if IMG_GET_RENDER:
                    obs = env.render(mode='rgb_array')
                ot = obs_preproc(obs)
            
                hx, cx = torch.zeros([1,hidden_dim]),torch.zeros([1,hidden_dim]) 
                copy_hx, copy_cx =torch.zeros([1,hidden_dim]),torch.zeros([1,hidden_dim]) 
                prev_hx, prev_cx =torch.zeros([1,hidden_dim]),torch.zeros([1,hidden_dim]) 
             
                prev_list = []
                local_buf = []
                total_reward=[]

                dt = False
                break 



            frame+=1
            with torch.no_grad():
                Qt, (hx,cx) = Q_main(ot,(hx,cx))
                #e greedy
                if random.random() >= policy_epsilon:
                    action =  torch.argmax(Qt,dim=1).item()
                else:
                    action = random.randint(0,feature_action-1)
            
            obs, rt,dt,_  = env.step(action)
            if IMG_GET_RENDER:
                obs=env.render(mode='rgb_array')
            ot_1 = obs_preproc(obs)
            total_reward.append(rt)
            local_buf.append([ot,action,rt,gamma_t])
            
            ot = ot_1
            if frame % def_actor_update_step == 0:
                Q_main.load_state_dict(shared_state["Q_state"])
        
        prev_list.extend(local_buf)
        if len(prev_list) > burn_in_length+n_step:
            for s in range( len(prev_list) ,sequences_length):
                prev_list.append([ot,action,0.,0.])
            state, action, reward,gamma = map(np.stack,zip(*prev_list))
            while shared_queue.qsize() > 50:
                print(f'#{rank} actor sleep')
                time.sleep(1)
            shared_queue.put([state,action,reward,gamma,prev_hx,prev_cx])
        
        prev_hx, prev_cx = copy_hx.clone(), copy_cx.clone()
        copy_hx, copy_cx = hx.clone(), cx.clone()
        prev_list = local_buf
        local_buf = []

        
def h_func(x):
    epsilon= 10e-2
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
def h_inv_func(x):
    epsilon= 10e-2
    return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
    
def learner_process(rank , shared_state, shared_queue, max_frame ,dev ):
    print(f'#{rank} learner process start ')

    Q_main = Duelling_LSTM_DQN(feature_state, feature_action).to(dev)
    Q_target = Duelling_LSTM_DQN(feature_state, feature_action).to(dev)
    Q_main.load_state_dict(shared_state["Q_state"])
    Q_target.load_state_dict(shared_state["Q_state"])
    
    
    value_optimizer  = optim.Adam(Q_main.parameters(),  lr=def_lr)
    global_buf = ReplayBuffer(def_global_buf_maxlen)

    frame = 0
    while len(global_buf) < train_start_size:
        print(f'\r g_buf len :{len(global_buf)}/{train_start_size}',end='\r')
        global_buf.append(shared_queue.get())
    
    
    while frame<max_frame:
        time.sleep(learner_frame_interval)

        for i in range(shared_queue.qsize()):
            global_buf.append(shared_queue.get())
        frame+=1

        batch = global_buf.sample(batch_size)

        state, action, reward, gamma, copy_hx, copy_cx = map(np.stack, zip(*batch))
        st = torch.from_numpy(state).reshape((sequences_length,batch_size)+feature_state).float().to(dev)
        at = torch.from_numpy(action).reshape((sequences_length,batch_size)).long().to(dev)
        rt = torch.from_numpy(reward).reshape((sequences_length,batch_size)).float().to(dev)
        gamt = torch.from_numpy(gamma).reshape((sequences_length,batch_size)).float().to(dev)
        
        hx_m = torch.from_numpy(copy_hx).reshape((batch_size,hidden_dim)).to(dev)
        cx_m = torch.from_numpy(copy_cx).reshape((batch_size,hidden_dim)).to(dev)
        
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
            target_Q_v, (hx_double_t,cx_double_t) = Q_target(st[i], (hx_double_t, cx_double_t))
            a_star = torch.argmax(target_Q_v, dim =1)
            
            main_Q_v , (hx_double_m,cx_double_m) = Q_main(st[i], (hx_double_m, cx_double_m))
            Q_tilda.append(main_Q_v.gather(1,a_star.view(batch_size,-1)))
            

        loss = torch.zeros([1]).to(dev)
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
        print(f'\r#{rank} frame:{frame:5d} loss:{loss.item()} g_buf_size:{len(global_buf):6d}/{def_global_buf_maxlen}',end='\r')


        if frame%def_learner_update_step ==0:
            tau = def_soft_update_tau
            for target_param, param in zip(Q_target.parameters(),Q_main.parameters()):
                target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau )
            state = dict()
            for k,v in Q_main.state_dict().items():
                state[k] = v.cpu()

            shared_state["Q_state"] = Q_main.state_dict()

        

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    dev_cpu = torch.device('cpu')
    dev_gpu = torch.device('cuda' if use_cuda else 'cpu')
    print(dev_cpu, dev_gpu)

    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)

    manager = mp.Manager()
    shared_state = manager.dict()
    shared_queue = manager.Queue()
    shared_state["Q_state"] = Q_main.state_dict()
    
    if USE_MP == False:
        actor_process(0, shared_state, shared_queue,100, dev_cpu, num_processes)
        learner_process(0, shared_state, shared_queue,2, dev_gpu)
    else: 
        learner_procs = mp.Process(target=learner_process, args=(999, shared_state, shared_queue,100000,dev_gpu,))
        learner_procs.start()
        
        actor_procs = []
        for i in range(num_processes):
            print(i)
            actor_proc = mp.Process(target=actor_process, args=(i, shared_state, shared_queue,1000000,dev_cpu,num_processes))
            actor_proc.start()
            actor_procs.append(actor_proc)
        for act in actor_procs:
            act.join()    
        learner_procs.join()
    






