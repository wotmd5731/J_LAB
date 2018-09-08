import numpy as np
import pandas as pd
from preprocess import data_pre_pro_walk_pandas_multikey


class stock_env():
    action_space=3
    observation_space=7

    def __init__(self,datapath,name_list,main_name):
        self.df = data_pre_pro_walk_pandas_multikey(datapath,name_list)
        self.last_date = self.df.iloc[-1].name
        self.used_buy = False
        self.start_date = '2017-01-03'
        self.today_date = self.start_date 
        self.buy_dataframe =[] 
        self.main_name = main_name
        self.fees_rate = 0.01
        self.observation=[]


    def _next_date(self,today):
        nn = (self.df.index==today).argmax()+1
        while True:
            if self.df.iloc[nn].name!=today:
                break
            nn +=1
        return self.df.iloc[nn].name
    def _find_data_from_date(self, date):
        return self.df.loc[date].drop_duplicates(['symbol'],keep='last').sort_values('symbol')

    def _find_symbol_value(self,obs,name=0,typ='close'):
        if name==0:
            name=self.main_name
        return obs[obs.symbol==name][typ].values

    def reset(self):
        self.used_buy=False
        self.today_date=self.start_date
        reward =0 
        done =0 
        info =0
        self.observation = self._find_data_from_date(self.today_date)
        self.buy_dataframe = self.observation
        return self.observation,reward,done,info

    def step(self, action):
        reward=0
        done=0
        info=0
        if action == 0:
            pass
        elif action ==1 and self.used_buy == False:
            self.used_buy=True
            self.buy_dataframe = self.observation
            reward = -1*self._find_symbol_value(self.observation)*self.fees_rate
        elif action ==2 and self.used_buy == True:
            self.used_buy=False
            reward= (1-self.fees_rate)*self._find_symbol_value(self.observation)\
                    -self._find_symbol_value(self.buy_dataframe)
        else:
            print("action_buy must be called before sell")
            return -1,-1,-1,-1
        self.today_date = self._next_date(self.today_date)
        self.observation=self._find_data_from_date(self.today_date)
        return self.observation,reward,done,info

    def render(self):
        pass

env = stock_env('2017data',['Z','FAX','FI'],'Z')
env.reset()


