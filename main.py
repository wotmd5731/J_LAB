# -*- coding: utf-8 -*-

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

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
    
data = data_preprocess('2017data')
sdata = sorted(data, key=lambda x: time.mktime(time.strptime(x[1],"%Y-%m-%d")))
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