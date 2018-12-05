import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

init_w = 3e-3
init_b = 3e-4

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNet(nn.Module):
    def __init__(self, dev, config):
        super(ActorNet, self).__init__()
        obs_size = config['obs_space']
        n_actions = config['action_space'][-1]
        
        self.use_cnn = config['use_cnn']
        if self.use_cnn:
            self.front_0 = nn.Conv2d(obs_size[1], 16, 8, stride=4)
            self.front_1 = nn.Conv2d(16*2, 32, 4, stride=2)
            self.front_2 = nn.Conv2d(32*2, 32, 3, stride=1)
            self.front_3 = nn.Linear(in_features=7*7*32*2, out_features=128)
        else:
            self.front_0 = nn.Linear(obs_size[1] , 128)
            self.front_1 = nn.Linear(in_features=128*2, out_features=128)
        
        
        self.policy_l0 = nn.Linear(128 , 32)
        self.policy_l1 = nn.Linear(32*2 , n_actions)
        
        
        
#        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
#        self.l2.weight_ih.data = fanin_init(self.l2.weight_ih.data.size())
#        self.l2.weight_hh.data = fanin_init(self.l2.weight_hh.data.size())
#        self.l3.weight.data.uniform_(-init_w, init_w)
#        self.l3.bias.data.fill_(init_b)


        self.dev = dev
        

    def __call__(self, xx):
        batch = xx.size(0)
        if self.use_cnn:
            xx = F.relu(self.front_0(xx))
            xx = F.relu(self.front_1(torch.cat([xx,-xx],dim=1)))
            xx = F.relu(self.front_2(torch.cat([xx,-xx],dim=1)))
            xx = F.relu(self.front_3(torch.cat([xx,-xx],dim=1).view(batch,-1)))
        else:
            xx = F.relu(self.front_0(xx))
            xx = F.relu(self.front_1(torch.cat([xx,-xx],dim=1)))
        
        
        policy = F.relu(self.policy_l0( xx ))
        policy = torch.sigmoid(self.policy_l1(torch.cat([policy,-policy],dim=1)))
        
        return policy
    
#        x = self.front(x)
#        if self.hx is None: # 200
#            self.hx = torch.zeros((x.size()[0] ,128)).to(self.dev)
#            self.cx = torch.zeros((x.size()[0] ,128)).to(self.dev)
#        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
#        
#        v = self.value(self.hx)
#        a = self.advantage(self.hx)
#        
#        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
#        
#        x = torch.tanh(self.hx)
#        x = torch.tanh(self.l3(x))
#        return x

    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return torch.zeros((1 ,128)), torch.zeros((1 ,128))
        else:
            return self.hx.clone().detach().cpu(), self.cx.clone().detach().cpu()

class CriticNet(nn.Module): # 400-300
    def __init__(self, dev,config):
        super(CriticNet, self).__init__()
        obs_size = config['obs_space']
        n_actions = config['action_space'][-1]
        
        
        self.use_cnn = config['use_cnn']
        if self.use_cnn:
            self.front_0 = nn.Conv2d(obs_size[1], 16, 8, stride=4)
            self.front_1 = nn.Conv2d(16*2, 32, 4, stride=2)
            self.front_2 = nn.Conv2d(32*2, 32, 3, stride=1)
            self.front_3 = nn.Linear(in_features=7*7*32*2, out_features=128)
        else:
            self.front_0 = nn.Linear(obs_size[1] , 128)
            self.front_1 = nn.Linear(in_features=128*2, out_features=128)
        
        
        self.value_l0 = nn.Linear(128 , 32)
        self.value_l1 = nn.Linear(32*2 , 1)
        
        self.adv_l0 = nn.Linear(128 , 32)
        self.adv_l1 = nn.Linear(32*2 , n_actions)
        
#        self.l1 = nn.Linear(in_features=obs_size + n_actions, out_features=128)
#        self.l2 = nn.LSTMCell(128, 128)
#        self.l3 = nn.Linear(in_features=128, out_features=n_actions)

#        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
#        self.l2.weight_ih.data = fanin_init(self.l2.weight_ih.data.size())
#        self.l2.weight_hh.data = fanin_init(self.l2.weight_hh.data.size())
#        self.l3.weight.data.uniform_(-init_w, init_w)
#        self.l3.bias.data.fill_(init_b)

        self.hx = None
        self.cx = None

        self.dev = dev

    def __call__(self, xx, a):
        batch = xx.size(0)
        if self.use_cnn:
            xx = F.relu(self.front_0(xx))
            xx = F.relu(self.front_1(torch.cat([xx,-xx],dim=1)))
            xx = F.relu(self.front_2(torch.cat([xx,-xx],dim=1)))
            xx = F.relu(self.front_3(torch.cat([xx,-xx],dim=1).view(batch,-1)))
        else:
            xx = F.relu(self.front_0(xx))
            xx= F.relu(self.front_1(torch.cat([xx,-xx],dim=1)))
        
        val = F.relu(self.value_l0( xx ))
        val = self.value_l1(torch.cat([val,-val],dim=1))
        
        adv = F.relu(self.adv_l0(torch.cat([ xx],dim=1) ))
        adv = self.adv_l1(torch.cat([adv,-adv],dim=1))
        act_val = val+adv
        return act_val
        
        
#        x = torch.cat((x,a), 1)
#        x = torch.tanh(self.l1(x))
#        if self.hx is None: # 300
#            self.hx = torch.zeros((x.size()[0] ,128)).to(self.dev)
#            self.cx = torch.zeros((x.size()[0] ,128)).to(self.dev)
#        self.hx, self.cx = self.l2(x, (self.hx, self.cx))
#        x = torch.tanh(self.hx)
#        x = self.l3(self.hx)
#        return x

    def reset_state(self):
        self.hx = None
        self.cx = None

    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx

    def get_state(self):
        if self.hx is None:
            return torch.zeros((1 ,128)), torch.zeros((1 ,128))
        else:
            return self.hx.clone().detach().cpu(), self.cx.clone().detach().cpu()