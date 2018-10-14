
import logging
import os
import settings
import data_manager
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import torch.nn.functional as F
from collections import deque
from torch.distributions import Dirichlet, Normal, kl_divergence, Categorical, Uniform

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs_pend')

import random
import gym

import visdom
vis = visdom.Visdom()

class env_stock():
    def __init__(self):
        stock_code = '005930'
        start_date = '2010-03-01'
        end_date = '2015-03-04'

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


        chart_data['data']= pd.to_datetime(chart_data.date).astype(np.int64)/1000000
        data = torch.from_numpy(chart_data.values)

        self.data = torch.stack([data[:,0],data[:,4],data[:,5]],dim=1).float()

        self.data = self.data - self.data.mean(dim=0)
        self.data = self.data/self.data.std(0)

        self.count_max= self.data.size(0)

    def step(self,action):
        #continuouse action space
        quantize = 100
        a_t = round((self.prev_action - action)*quantize)
        r_t = d_t= 0

        if a_t >=0:
            self.pocket += self.data[self.count,1]*a_t
        else:
            self.pocket += self.data[self.count,1]*a_t

        self.prev_action = action

        if self.count+1 == self.count_max:
            self.pocket += self.data[self.count,1]*(self.prev_action*quantize)
            d_t = 1
            r_t = self.pocket 

        else:
            self.count +=1

        return self.data[self.count].view(1,-1),r_t ,d_t
        pass


env = env_stock()
env.reset()

dev = torch.device('cpu')
print(dev)
from DDPG_network import Agent

agent = Agent(nb_states=3,nb_action=1,mem_size=10000,dev=dev)
agent.train()

for episode in range(1000):
    testset = True if (episode+5)%100 == 0 else False
    ite = 0
    test_epi = 0
    train_epi = 0
    max_v_l = -9999
    max_p_l = -9999
    mem = []
    s_t = env.reset()
    total_reward = 0
    eps = 0 if testset else 0.3

    for T in range(10000):
        a_t = agent.get_action(s_t,eps=eps).clamp(min=0,max=1)
        s_t_1,r_t,done = env.step(a_t.item())
        mem.append([s_t,a_t,r_t,s_t_1,done])
        total_reward += r_t
        s_t = s_t_1

        if T%100 == 0 and episode>18 and eps>0.001:
            v_l , p_l = agent.update_policy()
            max_v_l = max(max_v_l,v_l)
            max_p_l = max(max_p_l,p_l)

            ite+=1
        if done:
            print('\r episode :{}  maxT:{} , total_reward:{} max_v_l:{} max_p_l:{}'.format(episode,T,total_reward,max_v_l,max_p_l,end='\r',flush=True)
        
            agent.mem_append(mem)
            break


        
