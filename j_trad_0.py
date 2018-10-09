# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:31:52 2018

@author: JAE
"""

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

#name = 'first_test'  # 삼성전자

# 로그 기록
#log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % name)
#timestr = settings.get_time_str()
#if not os.path.exists('logs/%s' % name):
#    os.makedirs('logs/%s' % name)
#file_handler = logging.FileHandler(filename=os.path.join(
#    log_dir, "%s_%s.log" % (name, timestr)), encoding='utf-8')
#stream_handler = logging.StreamHandler()
#file_handler.setLevel(logging.DEBUG)
##stream_handler.setLevel(logging.INFO)
#stream_handler.setLevel(logging.DEBUG)
#logging.basicConfig(format="%(message)s",
#                    handlers=[file_handler, stream_handler], level=logging.DEBUG)



class env_stock():
    def __init__(self,stock_list,start,end):
                
        self.stock_list = stock_list
        self.start_date  = start
        self.end_date  = end
        
        data_base = []
        
        for stock_code in self.stock_list:
            # 주식 데이터 준비
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
            chart_data['date'] = pd.to_datetime(chart_data.date).values.astype(np.int64)
            
            #차트 index축을 date 로변경
        #    chart_data.set_index('date', inplace=True)
            
            chart_data = torch.from_numpy(chart_data.values)
            data_base.append(chart_data)
        
        data_base = torch.cat(data_base,dim=1).float()
        scaled_data = (data_base-data_base.mean(dim=0))
        self.scaled_data = scaled_data/scaled_data.std()


        self.max_count  = self.scaled_data.size(0)
        
    def reset(self):
        self.record = 0
        self.action_list = [ 1,2]
        self.count = 0
        s_t = self.scaled_data[self.count]
        return s_t
    
    def step(self,a_t):
        s_t = 0
        r_t = d_t = 0
        if self.count == self.max_count-1:
            d_t = 1
            
        if a_t == 1 and not d_t and a_t in self.action_list:
            "buy"
            self.action_list.remove(1)
            
            self.record = self.scaled_data[self.count]
        elif a_t == 2 or (d_t and self.record != 0):
            'sell'
            self.action_list.append(1)
            r_t = self.scaled_data[self.count] - self.record
            self.record = 0
            
            r_t = r_t[4].item() # 첫번째 stock code의 close price기준.
        self.count+=1
        s_t = self.scaled_data[self.count]
            
            
        return s_t, r_t, d_t,0
    
    
    




env = env_stock(['005930','015760','035420'],'2017-10-01','2017-12-31')
env.reset()
env.step(0)

#train
for episode in range(100):
    s_t = env.reset()
    for step in range(20000):
        #get_action from agent
        a_t = agent.get_action(s_t)
        
        s_t_1, r_t, d_t,_ = env.step(a_t)
        
        
        s_t = s_t_1 
        if d_t:
            break
            #done
            
        
        
        

#test





# 학습 데이터 분리
#features_training_data = [
#    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
#    'close_lastclose_ratio', 'volume_lastvolume_ratio',
#    'close_ma5_ratio', 'volume_ma5_ratio',
#    'close_ma10_ratio', 'volume_ma10_ratio',
#    'close_ma20_ratio', 'volume_ma20_ratio',
#    'close_ma60_ratio', 'volume_ma60_ratio',
#    'close_ma120_ratio', 'volume_ma120_ratio'
#]
#training_data = training_data[features_training_data]
