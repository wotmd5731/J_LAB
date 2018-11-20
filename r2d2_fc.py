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

import os
os.system('cls')

def move (y, x):
    print("\033[%d;%dH" % (y, x))
    



USE_MP = True
#USE_MP = False

batch_size = 64
burn_in_length = 4
sequences_length = 12

hidden_dim = 128
n_step = 2
def_gamma = 0.997
def_global_buf_maxlen = 100000
def_lr = 0.001
def_soft_update_tau = 0.7
def_learner_update_step = 50
def_actor_update_step = 500
train_start_size = 1000
num_processes = 4
learner_frame_interval = 0#sec
topk_process = 2


test = 0
if test == 1:
#    USE_MP = False
    batch_size = 4
    train_start_size = 10
    num_processes = 3
    topk_process=1
    
    
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
        
#        self.lstm = torch.nn.LSTMCell(input_size=self.lstm_in_size , hidden_size=hidden_dim)
#        input of shape (batch, input_size): tensor containing input features
#        hidden of shape (batch, hidden_size)
        
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( hidden_dim, hidden_dim),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(hidden_dim, 1)
        self.advantage = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.front(x)
        x = x.view(-1,self.lstm_in_size)
#        hidden = self.lstm(x, hidden)
#        x=hidden[0]
        
#        output of shape (seq_len, batch, num_directions * hidden_size)
        
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return action_value


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


def actor_process(rank, shared_state, shared_queue, max_frame , dev, num_processes, schedule_dict):
    random.seed(time.time())
    
    Q_main = Duelling_LSTM_DQN(feature_state, feature_action)
    Q_main.load_state_dict(shared_state["Q_state"])
    
    env = gym.make(GAME_NAME)
    policy_epsilon = 0.05**((rank+1)/num_processes)
    win_r = vis.line(Y=torch.Tensor([0]), opts=dict(title ='reward'+str(policy_epsilon)))
    

    
    print(f'#{rank} actor process start p:{policy_epsilon}')

    action = 0
    gamma_t = def_gamma
    frame = 0
    ttime = time.time()

    total_reward = []
    prev_list = []
    local_buf = []
    dt = True
    win_i = 0
    
    while frame < max_frame:
        schedule_dict[str(rank)+'frame']=frame
        while schedule_dict[str(rank)+'sleep'] :
            time.sleep(0.5)

        for seq in range(501):
            if dt==True :
                win_i += 1
#                if win_i%10 == 0:
                win_r = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([sum(total_reward)]), win= win_r , update ='append')
                    
                print(f'#{rank} frame:{frame:5d} total_reward: {sum(total_reward)}, step:{len(total_reward)}, time:{time.time()-ttime}')
                ttime = time.time()
                #env reset
                obs = env.reset()
                if IMG_GET_RENDER:
                    obs = env.render(mode='rgb_array')
                ot = obs_preproc(obs)
            
             
#                prev_list = []
                local_buf = []
                total_reward=[]
                gamma_t = def_gamma
                dt = False
                break 



            frame+=1
            with torch.no_grad():
                Qt = Q_main(ot)
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
#            local_buf.append([ot,action,rt,gamma_t,ot_1])
            if dt:
                gamma_t = 0
            shared_queue.put([ot,action,rt,gamma_t,ot_1])
            
            
            ot = ot_1
            if frame % def_actor_update_step == 0:
                Q_main.load_state_dict(shared_state["Q_state"])
        
#        prev_list.extend(local_buf)
#        if len(prev_list) > burn_in_length+n_step:
#            for s in range( len(prev_list) ,sequences_length):
#                prev_list.append([ot,action,0.,0.])
#        state, action, reward,gamma,state_next = map(np.stack,zip(*local_buf))
#        while shared_queue.qsize() > 50:
#            print(f'#{rank} actor sleep')
#            time.sleep(1)
#        shared_queue.put([state,action,reward,gamma,state_next])
        
#        prev_hx, prev_cx = copy_hx.clone(), copy_cx.clone()
#        copy_hx, copy_cx = hx.clone(), cx.clone()
#        prev_list = local_buf

        
def h_func(x):
    epsilon= 10e-2
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
def h_inv_func(x):
    epsilon= 10e-2
    return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
    
def learner_process(rank , shared_state, shared_queue, max_frame ,dev,schedule_dict ):
#    print(f'#{rank} learner process start ')

    win_r = vis.line(Y=torch.Tensor([0]), opts=dict(title ='loss'))

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
        schedule_dict[str(rank)+'frame']=frame*5
        while schedule_dict[str(rank)+'sleep'] :
            time.sleep(0.5)
#        time.sleep(learner_frame_interval)

        for i in range(shared_queue.qsize()):
            global_buf.append(shared_queue.get())
        frame+=1

        batch = global_buf.sample(batch_size)

        state, action, reward, gamma, next_state = map(np.stack, zip(*batch))
        st = torch.from_numpy(state).reshape(tuple([batch_size])+feature_state).float().to(dev)
        at = torch.from_numpy(action).reshape(tuple([batch_size])).long().to(dev)
        rt = torch.from_numpy(reward).reshape(tuple([batch_size])).float().to(dev)
        gamt = torch.from_numpy(gamma).reshape(tuple([batch_size])).float().to(dev)
        st_1 = torch.from_numpy(next_state).reshape(tuple([batch_size])+feature_state).float().to(dev)



 
        
#        loss = torch.zeros([1]).to(dev)
        
        Qv = Q_main(st).gather(1,at.view(batch_size,-1)).view(-1)
        with torch.no_grad():
            Qt = Q_target(st_1).max(1)[0] * gamt + rt
        
        loss = F.mse_loss(Qv, Qt)   
#        print(loss)
        
        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()
        print(f'#{rank} frame:{frame:5d} loss:{loss.item()} g_buf_size:{len(global_buf):6d}/{def_global_buf_maxlen}')
#        if frame%10==0:
        win_r = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([loss.item()]), win= win_r , update ='append')
                

        if frame%def_learner_update_step ==0:
            tau = def_soft_update_tau
            for target_param, param in zip(Q_target.parameters(),Q_main.parameters()):
                target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau )
            state = dict()
            for k,v in Q_main.state_dict().items():
                state[k] = v.cpu()

            shared_state["Q_state"] = state

        

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

    schedule_dict = manager.dict()
    for i in range(num_processes):
        schedule_dict[str(i)+'frame']=0
        schedule_dict[str(i)+'sleep']=False


    
    if USE_MP == False:
        actor_process(1, shared_state, shared_queue,100, dev_cpu, num_processes, schedule_dict)
        learner_process(0, shared_state, shared_queue,2, dev_gpu)
    else: 
        learner_procs = mp.Process(target=learner_process, args=(0, shared_state, shared_queue,100000,dev_gpu,schedule_dict))
        learner_procs.start()
        
        actor_procs = []
        for i in range(1, num_processes):
#            print(i)
            actor_proc = mp.Process(target=actor_process, args=(i, shared_state, shared_queue,1000000,dev_cpu,num_processes, schedule_dict))
            actor_proc.start()
            actor_procs.append(actor_proc)


        frame = [0 for i in range(num_processes)]
        while True:
            time.sleep(0.5)
            for i in range(num_processes):
                frame[i] = schedule_dict[str(i)+'frame']
            
            move(3,0)
            print(frame)
            if max(frame) - min(frame) > 100:
                idx = torch.LongTensor(frame).topk(topk_process)[1]
                for i in range(num_processes):
                    if i in idx:
                        schedule_dict[str(i)+'sleep']=True
                    else:
                        schedule_dict[str(i)+'sleep']=False
                        

        for act in actor_procs:
            act.join()    
        learner_procs.join()
    






