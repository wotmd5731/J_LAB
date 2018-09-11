# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:31:53 2018

@author: JAE
"""

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')


import numpy as np

ite = 0
x = np.linspace(0,1,100)
for ii in x:
    y = np.sin(ii*ii*3)
    writer.add_scalar('yy',y,ite)
    ite +=1