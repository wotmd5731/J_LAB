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
        total_data = list(set(total_data))
    print(total_data)
    
data_preprocess('2017data')
#f = open(data_path,'r')


#f = open('data.csv','r',encoding='utf-8')
#rdr = csv.reader(f)
#data = []
#for line in rdr:
#    data.append(line[-7:])
##    print(line[-7:1])
#f.close()