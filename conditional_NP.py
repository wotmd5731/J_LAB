import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, MultivariateNormal

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections

class DeterministicEncoder(nn.Module):
    def __init__(sefl,netsize):
        "netsize = [start,128,128,128,128]"
        super(DeterministicEncoder,self).__init__()
        
        self.net = nn.ModuleList([nn.Linear(netsize[i],netsize[i+1]) for i in range(len(netsize)-1)])


    def forward(self,context_x, context_y):
        encoder_input = torch.cat([context_x,context_y],-1)
        batch_size, num_context_points, filter_size = encoder_input.size()
        hidden = torch.reshape(encoder_input,(-1,filter_size))

        for i in range(len(self.net)-1):
            hidden = F.relu(self.net[i](hidden))
        else :
            hid = self.net[i+1](hidden)

        hid = torch.reshape(hid,(batch_size,num_context_points,-1))
        representation = torch.mean(hid,1)
        return representation


class DeterministicDecoder(nn.Module):
    def __init__(sefl,netsize):
        "netsize = [start,128,128,2]"
        super(DeterministicDecoder,self).__init__()
        
        self.net = nn.ModuleList([nn.Linear(netsize[i],netsize[i+1]) for i in range(len(netsize)-1)])


    def forward(self,representation, target_x):
        num_total_points = target_x.size()[1]

        rep = torch.stack([representation for i in range(num_context_points)],dim=1)
        decoder_input = torch.cat([rep,target_x],dim=-1)
        batch_size,_, filter_size = decoder_input.size()
        hidden = torch.reshape(decoder_input,(-1,filter_size))

        for i in range(len(self.net)-1):
            hidden = F.relu(self.net[i](hidden))
        else :
            hid = self.net[i+1](hidden)

        hid = torch.reshape(hid,(batch_size,num_total_points,-1))
        mu, log_sigma = torch.split(hid,1,dim=-1)
        sigma = 0.1+0.9*F.softplus(log_sigma)

        batch_size = sigma.size()[0]
        sigma_diag=torch.stack([torch.diagflat(sigma[bat]) for bat in range(batch_size)],0)

        dist = MultivariateNormal(mu.squeeze(-1), scale_tril=sigma_diag)
        return dist, mu, sigma


dev = torch.device('cuda') if torch.cuda.is_available() and False else torch.device('cpu')
print(dev)

enc = DeterministicEncoder([2,128,128,128,128]).to(dev)
dec = DeterministicDecoder([128+1,128,128,2]).to(dev)

optimizer = torch.optim.Adam([
    {'params':enc.parameters()},
    {'params':dec.parameters()}]
    ,lr=1e-4)

TRAINING_ITERATIONS = int(2e5)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = 100


# dataset_train ~ test  object  gen


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


import time
prev_time = time.time()
def print_time(val=0):
    global prev_time
    print(val ,' time ',time.time()-prev_time)
    prev_time=time.time()



load_model(enc,dec,'cnp_back')

for it in range(TRAINING_ITERATIONS):
    train = sess.run(data_train)
    context_x = torch.from_numpy(train.query[0][0]).to(dev)
    context_y = torch.from_numpy(train.query[0][1]).to(dev)
    target_x = torch.from_numpy(train.query[1]).to(dev)
    target_y = torch.from_numpy(train.target_y).to(dev)
    
    optimizer.zero_grad()
    rep = enc(context_x,context_y)
    dist, mu, sigma = dec(rep,target_x)
    log_p = dist.log_prob(target_y.squeeze(-1))
    loss = -log_p.mean()
    loss.backward()
    optimizer.step()

    if it%PLOT_AFTER == 0:
        test = sess.run(data_test)
        context_x = torch.from_numpy(test.query[0][0]).to(dev)
        context_y = torch.from_numpy(test.query[0][1]).to(dev)
        target_x = torch.from_numpy(test.query[1]).to(dev)
        target_y = torch.from_numpy(test.target_y).to(dev)
        
        rep = enc(context_x,context_y)
        dist, mu, sigma = dec(rep,target_x)
        log_p = dist.log_prob(target_y.squeeze(-1))
        loss = -log_p.mean()
        
        pred_y = mu
        var = sigma
        
        plot_functions(~~)

        print('Iterations {}, loss:{}'.format(it,loss))

save_model(enc,dec,'cnp_back')





