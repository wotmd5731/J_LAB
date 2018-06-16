# -*- coding: utf-8 -*-

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import datetime
from dateutil import parser
import os
import csv
import matplotlib.pyplot as plt

def data_preprocess(dir_path):
    dir_list = os.listdir(dir_path)
    total_data = []
    for dir_csv in dir_list:
        total_path = dir_path+'/'+dir_csv+'/prices.csv'
#        print(total_path)
        file = open(total_path,'r')
        rdr = csv.reader(file)
#        for d in rdr:
#            if 'FAX' in d[0]:
#                total_data.append(d)
#                break
        [total_data.append(d) for d in rdr if 'FAX' in d[0]]
#        total_data = list(set(total_data))
    
#    print(total_data)
    return total_data
    

def data_pre_pro_walk(dir_path, key):
    total_data = []
    for (paths, dirs, files) in os.walk(dir_path):
        for fs in files:
            if fs == 'prices.csv':
#                print(paths,fs)
                with open(paths+'/'+fs,'r') as file:
                    rdr = csv.reader(file)
#                    [total_data.append(d) for d in rdr if key in d[0]]
                    for da in [d for d in rdr if key in d[0]]:
                        da.extend([parser.parse(da[1]).weekday()])
                        total_data.append(da)
#                        print(da)
                    
    np_sdata = np.array(total_data)
    #np_sdata[:,1] is means the date
    # following command applies unique to the date!
    # unique  is  always sorted
    uni_np, indic = np.unique(np_sdata[:,1],return_index=True)
    
#    print(np_sdata[indic])
#    print(uni_np)
#sdata_sorted = sorted(sdata,key=lambda x: time.mktime(time.strptime(x[1],"%Y-%m-%d")))
    return np_sdata[indic]
#data = data_preprocess('2017data')
#sdata = sorted(data, key=lambda x: time.mktime(time.strptime(x[1],"%Y-%m-%d")))


sdata = data_pre_pro_walk('2017data','FAX')




#[da for i, da in reversed(enumerate(sdata)) if sdata[i][1]!=sdata[i+1][1]  ]

#sdata = sorted(data, key=lambda kk: data[1])
#
#timestamps = ['2011-06-2', '2011-08-05', '2011-02-04', '2010-1-14', '2010-12-13', '2010-1-12', '2010-2-11', '2010-2-07', '2010-12-02', '2011-11-30', '2010-11-26', '2010-11-23', '2010-11-22', '2010-11-16']
#timestamps.sort(key=lambda x: time.mktime(time.strptime(x,"%Y-%m-%d")))


#f = open(data_path,'r')
 

#f = open('data.csv','r',encoding='utf-8')
#rdr = csv.reader(f)
#data = []
#for line in rdr:
#    data.append(line[-7:])
##    print(line[-7:1])
#f.close()