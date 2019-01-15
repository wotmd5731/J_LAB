
import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
import time
import os
from torch.distributions import Normal

vis = visdom.Visdom(port = 8097)
import torch.multiprocessing as mp
#win_4 = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'))


import time

ttime= time.time()
def time_check(num=0):
    global ttime
    print(f'{num} time:{time.time()-ttime}')
    ttime = time.time()

"""
+double
+dueling
+episodic mem
+nstep
+per
+image version
+frame_stack
+RND
+lstm
"""
mu = 0
sigma = 1
vt = 0
POPART_BETA = 0.003

MAX_ACCESS = 1000
SOFT_TAU = 0.05
COUNT_MIN = 1000
V=0
P=1
Q=2
TV=3
TP=4
TQ=5
ALPHA=0.2

max_shared_q_size = 5
frame_stack = 1
n_step = 5
PER_alpha = 0.9  # 0 is uniform per
count_episode = False
RND_const = 0
start_frame = 50
num_frames = 30000
batch_size =32
vis_render=True
EPS_CONST = 1
a_lr = 0.0001
c_lr = 0.001
v_lr = 0.001

rnd_lr = 0.00001
burn_in_len = 5
mem_size = 20000
seq_len = 7
env_id = 'CartPole-v0'

#env = gym.make(env_id)
cnn_enable=True
s_dim = 1*frame_stack
a_dim = 2
state_shape = (1,1,84,84)
import torchvision
togray = torchvision.transforms.Grayscale()
toten = torchvision.transforms.ToTensor()
resize = torchvision.transforms.Resize((84,84))
topil = torchvision.transforms.ToPILImage()
def obs_preproc(x):
    xten = toten(togray(resize(topil(x))))
    return xten.reshape(state_shape)

class env_cover():
    def __init__(self,env_id):
        self.env = gym.make(env_id)
    def reset(self):
        ss = self.env.reset()
        ss = np.delete(ss,[1,3])
        return torch.from_numpy(ss).float().view(1,s_dim).to(dev)
    #return obs_preproc(env.render(mode='rgb_array')).to(dev)
    def step(self,act):
        ss,rr,dd,_ = self.env.step(act)
        ss = np.delete(ss,[1,3])
        return torch.from_numpy(ss).float().view(1,s_dim).to(dev),rr,dd,0
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()


cnn_enable = False
vis_render=False
#s_dim = 2
s_dim = 2
state_shape = (1,1,s_dim)
#a_dim = 3

#env_id = 'MountainCar-v0'





use_cuda = False
use_cuda = torch.cuda.is_available()
dev = torch.device('cuda' if use_cuda else 'cpu')
print(dev)

import torch.utils.data

from collections import deque

class ReplayBuffer():
    def __init__(self,capacity, models, shared_state):
        self.win_bar = vis.bar(X=torch.rand([10]))
        self.win_bar_td = vis.bar(X=torch.rand([10]))

        self.count = 0
        self.capacity = capacity
        self.buffer = deque(maxlen= capacity)
        self.models = models
        self.shared_state = shared_state
    def push(self, data ):
        
#        [[state ,action,reward,gamma,ireward,igamma ],state_mem]
        with torch.no_grad():
            state   = data[0].to(dev)
            action  = data[1].to(dev)
            reward  = data[2].to(dev)
            gamma   = data[3].to(dev)
            ireward = data[4].to(dev)
            igamma  = data[5].to(dev)
            
            [self.models[i].reset_state() for i in range(4)]
            model_state = [self.models[i].get_state() for i in range(4)]
            
            
            b_len = state.size(0)
            
            loss_q1,loss_q2,loss_value,loss_policy, state_mem = calc_td(self.models,state,action,reward,gamma,ireward,igamma,
                               model_state, 
                               b_len-n_step, stored_state=True)
        
        
        
        
        self.count += data[0].size(0) if not count_episode else 1
        priority = []
        eta = 0.9
        td_loss = (loss_q1.sqrt()+loss_q2.sqrt()+loss_value.sqrt()).view(-1)
        for i in range(len(td_loss)-seq_len):
            p = (eta*td_loss[i:i+seq_len].max()+(1.-eta)*td_loss[i:i+seq_len].mean())**PER_alpha
            priority.append(p)
            
        priority = torch.stack(priority).view(-1)
#        td_loss_total = sum(priority)/len(priority)
        td_loss_total = priority.max()
        with self.shared_state["vis"].get_lock():
            vis.bar(X=td_loss.cpu().view(-1,1), win= self.win_bar_td, opts=dict(title='push td_loss'))
        self.buffer.append([data,td_loss,priority,td_loss_total,state_mem,0])
        while self.count > self.capacity:
            self.count -= self.buffer.popleft()[0][0].size(0)  if not count_episode else 1

    def sample(self,batch_size):
        weight = [self.buffer[i][3] for i in range(len(self.buffer))]
        batch_epi = list(torch.utils.data.WeightedRandomSampler(torch.stack(weight),batch_size, True))
        s = []
        for episode_idx in batch_epi:
            episode = self.buffer[episode_idx][0]
            priority = self.buffer[episode_idx][2]
            state_mem = self.buffer[episode_idx][4]
            
            ii = list(torch.utils.data.WeightedRandomSampler(priority , 1, True))[0]
         
            start = ii - burn_in_len if ii-burn_in_len>=0 else 0
            burn_state = episode[0][start:ii].to(dev)
            burn_action = episode[1][start:ii].to(dev)
            
            model_state = torch.cat(state_mem[start],0)
            
#            model_state = torch.cat([ahxcx,tahxcx,chxcx,tchxcx],0)
            
            state   =episode[0][ii:ii+seq_len+n_step]  
            action  =episode[1][ii:ii+seq_len+n_step]
            reward  =episode[2][ii:ii+seq_len+n_step]
            gamma   =episode[3][ii:ii+seq_len+n_step] 
            ireward =episode[4][ii:ii+seq_len+n_step]
            igamma  =episode[5][ii:ii+seq_len+n_step]
            
            s.append([episode_idx,ii,state,action,reward,gamma,ireward,igamma, model_state ,burn_state, burn_action])

        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,model_state, burn_state, burn_action = zip(*s)
        
        shape = (batch_size,-1)
        state   = torch.cat(state,1).to(dev)
        action   = torch.cat(action,1).to(dev)
        reward   = torch.cat(reward,1).to(dev)
        gamma   = torch.cat(gamma,1).to(dev)
        ireward   = torch.cat(ireward,1).to(dev)
        igamma   = torch.cat(igamma,1).to(dev)
        
        epi_idx = torch.LongTensor(epi_idx).reshape(shape).to(dev)
        seq_idx = torch.LongTensor(seq_idx).reshape(shape).to(dev)
        
        hxcx = torch.cat(model_state,0).reshape((batch_size,8,1,-1 )).to(dev)
#        mcx = torch.cat(model_state[1],0).reshape((batch_size,8,1,-1 )).to(dev)
#        thx = torch.cat(model_state[2],0).reshape((batch_size,8,1,-1 )).to(dev)
#        tcx = torch.cat(model_state[3],0).reshape((batch_size,8,1,-1 )).to(dev)
        
        return epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,hxcx, burn_state ,burn_action
    
    def mem_remove(self,idx):
        idx = reversed(idx.unique(sorted=True))
        if self.count < COUNT_MIN:
            return 0
        for epi_idx in idx:
            if self.buffer[epi_idx][5]>MAX_ACCESS:
                print(epi_idx.item(),end=' ,')
                self.count -= self.buffer[epi_idx][0][0].size(0) if not count_episode else 1
                del self.buffer[epi_idx]



    def priority_update(self,epi_idx,seq_idx,td_loss):
        td_array = self.buffer[epi_idx][1]
        self.buffer[epi_idx][5] +=1

#        priority = self.buffer[epi_idx][2]
#        total_priority = self.buffer[epi_idx][3]
        
        for i in range(seq_len):
            td_array[seq_idx+i] = td_loss[i]
#        for i in range(seq_len):
#            priority[seq_idx+i] = loss[i]
        
        start = seq_idx-seq_len 
        start = start if start>=0 else 0
        end = seq_idx+seq_len
        end = end if end<= len(td_array)-seq_len else len(td_array)-seq_len

        eta = 0.9
        for i in range(start, end):
            p = (eta*td_array[i:i+seq_len].max()+(1.-eta)*td_array[i:i+seq_len].mean())**PER_alpha
            self.buffer[epi_idx][2][i] = p.view(-1)
        self.buffer[epi_idx][3] = sum(self.buffer[epi_idx][2])/len(self.buffer[epi_idx][2])
        bar = []
        for i in range(len(self.buffer)):
            bar.append(self.buffer[i][3])

        with self.shared_state["vis"].get_lock():
            vis.bar(X=torch.stack(bar), win= self.win_bar, opts=dict(title='total priority'))
        
    def __len__(self):
        return self.count
    def __repr__(self):
        return '\rmem size: {}/{} ' .format(self.count, self.capacity)



class Flatten(nn.Module):
    def forward(self,inputs):
        return inputs.view(inputs.size(0),-1)

class QCritic(nn.Module):
    def __init__(self, num_inputs, num_action,  dev ):
        super(QCritic,self).__init__()
        if cnn_enable:
            size=7*7*64
            self.feature = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.PReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.PReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.PReLU(),
                    Flatten(),
                    nn.Linear(size,120),nn.PReLU(),
                    )
            self.feature_action = nn.Sequential(
                    nn.Linear(num_action,8),nn.PReLU(),
                    )
        else :
            self.feature = nn.Sequential(
                    nn.Linear(s_dim,120),nn.PReLU(),
                    )
            self.feature_action = nn.Sequential(
                    nn.Linear(num_action,8),nn.PReLU(),
                    )

        self.lstm_size = 128
        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        
#        self.advantage = nn.Sequential(
#                nn.Linear(self.lstm_size,128),nn.PReLU(),
#                nn.Linear(128,128),nn.PReLU(),
#                nn.Linear(128,num_outputs),
#                )
        self.value1 = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.value2 = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.last1 = self.value1[4]
        self.last2 = self.value2[4]

#        self.iadvantage = nn.Sequential(
#                nn.Linear(self.lstm_size,128),nn.PReLU(),
#                nn.Linear(128,128),nn.PReLU(),
#                nn.Linear(128,num_outputs),
#                )
        self.ivalue = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.hx = None
        self.cx = None

        self.dev = dev
        
    def forward(self,x,a,mu=0,sigma=1):
#        aa = torch.ones(x.size())*a
#        x = torch.cat([x,aa],dim=1)
        x = torch.cat([ self.feature(x), self.feature_action(a)],1)

        
        if self.hx is None: 
            self.hx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            self.cx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            
        self.hx, self.cx = self.lstm(x , (self.hx, self.cx))
        
        x= self.hx
        
        #adv = self.advantage(x)
        val1 = self.value1(x)*sigma + mu
        val2 = self.value2(x)*sigma + mu
        #iadv = self.iadvantage(x)
        ival = self.ivalue(x)
        
#        Q = val + adv - adv.mean()
#        iQ = ival + iadv - iadv.mean()
#        Qa = Q.argmax(1).view(-1,1)
#        iQa = iQ.argmax(1).view(-1,1)
        return val1,val2,ival,0

    
    def set_state(self, hxcx ):
        self.hx = hxcx[0]
        self.cx = hxcx[1]
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return torch.stack([torch.zeros((1 ,self.lstm_size)).to(self.dev), torch.zeros((1 ,self.lstm_size)).to(self.dev)],0)
        else:
            return torch.stack([self.hx.detach(), self.cx.detach()],0)
         
class VCritic(nn.Module):
    def __init__(self, num_inputs, num_action,  dev ):
        super(VCritic,self).__init__()
        if cnn_enable:
            size=7*7*64
            self.feature = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.PReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.PReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.PReLU(),
                    Flatten(),
                    nn.Linear(size,128),nn.PReLU(),
                    )
        else :
            self.feature = nn.Sequential(
                    nn.Linear(s_dim,128),nn.PReLU(),
                    )

        self.lstm_size = 128
        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        
#        self.advantage = nn.Sequential(
#                nn.Linear(self.lstm_size,128),nn.PReLU(),
#                nn.Linear(128,128),nn.PReLU(),
#                nn.Linear(128,num_outputs),
#                )
        self.value = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.last = self.value[4]

#        self.iadvantage = nn.Sequential(
#                nn.Linear(self.lstm_size,128),nn.PReLU(),
#                nn.Linear(128,128),nn.PReLU(),
#                nn.Linear(128,num_outputs),
#                )
        self.ivalue = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.hx = None
        self.cx = None

        self.dev = dev
        
    def forward(self,x,mu=0,sigma=1):
#        aa = torch.ones(x.size())*a
#        x = torch.cat([x,aa],dim=1)
        x = self.feature(x)

        
        if self.hx is None: 
            self.hx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            self.cx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            
        self.hx, self.cx = self.lstm(x , (self.hx, self.cx))
        
        x= self.hx
        
        #adv = self.advantage(x)
        val1 = self.value(x)*sigma + mu
        #iadv = self.iadvantage(x)
        ival = self.ivalue(x)
        
#        Q = val + adv - adv.mean()
#        iQ = ival + iadv - iadv.mean()
#        Qa = Q.argmax(1).view(-1,1)
#        iQa = iQ.argmax(1).view(-1,1)
        return val1,ival

    
    def set_state(self, hxcx ):
        self.hx = hxcx[0]
        self.cx = hxcx[1]
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return torch.stack([torch.zeros((1 ,self.lstm_size)).to(self.dev), torch.zeros((1 ,self.lstm_size)).to(self.dev)],0)
        else:
            return torch.stack([self.hx.detach(), self.cx.detach()],0)
         
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, dev ):
        super(Actor,self).__init__()
        if cnn_enable:
            size=7*7*64
            self.feature = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.PReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.PReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.PReLU(),
                    Flatten(),
                    nn.Linear(size,128),nn.PReLU(),
                    )
        else :
            self.feature = nn.Sequential(
                    nn.Linear(s_dim,128),nn.PReLU(),
                    )

        self.lstm_size = 128
        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        
        self.mean_net = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,num_outputs)
                )
        self.log_std_net = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,num_outputs)
                )
        self.epsilon = 1e-6
        self.hx = None
        self.cx = None

        self.dev = dev
        
    def forward(self,x, eval=False):
        
        x = self.feature(x)
        log_prob = None

        if self.hx is None: 
            self.hx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            self.cx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            
        self.hx, self.cx = self.lstm(x , (self.hx, self.cx))
        
        mu = self.mean_net(self.hx)
        log_std = self.log_std_net(self.hx).clamp(min=-20,max=2)
        std = torch.exp(log_std)
        normal = Normal(mu,std)
        z = normal.rsample()
        action = torch.tanh(z)
        if eval:
            log_prob = normal.log_prob(z) - torch.log(1-action.pow(2) + self.epsilon)
            log_prob = log_prob.sum(-1,keepdim=True)
        return action, log_prob, z, mu, log_std

    
    def set_state(self, hxcx ):
        self.hx = hxcx[0]
        self.cx = hxcx[1]
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return torch.stack([torch.zeros((1 ,self.lstm_size)).to(self.dev), torch.zeros((1 ,self.lstm_size)).to(self.dev)],0)
        else:
            return torch.stack([self.hx.detach(), self.cx.detach()],0)
        

class RND(nn.Module):
    def __init__(self,num_inputs):
        super(RND,self).__init__()
        if cnn_enable:
            size=7*7*64
            self.target = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.PReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.PReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.PReLU(),
                    Flatten(),
                    nn.Linear(size,128),nn.PReLU(),
                    nn.Linear(128,128),
                    )
            self.predictor = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.PReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.PReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.PReLU(),
                    Flatten(),
                    nn.Linear(size,128),nn.PReLU(),
                    nn.Linear(128,128),nn.PReLU(),
                    nn.Linear(128,128),
                    )
        else :
            self.target = nn.Sequential(
                    nn.Linear(s_dim,128),nn.PReLU(),
                    nn.Linear(128,128),
                    )
            self.predictor = nn.Sequential(
                    nn.Linear(s_dim,128),nn.PReLU(),
                    nn.Linear(128,128),nn.PReLU(),
                    nn.Linear(128,128),
                    )
            
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight,np.sqrt(2))
                m.bias.data.zero_()
        for param in self.target.parameters():
            param.requires_grad =False

    def forward(self, obs, next_obs):
        Tobs = torch.cat([obs,next_obs],dim=1)
        target_feature = self.target(Tobs)
        predict_feature = self.predictor(Tobs)
        return predict_feature*RND_const, target_feature*RND_const








def update_target(tar,cur):
    tar.load_state_dict(cur.state_dict())

def calc_td(models,state, action, reward,gamma,ireward,igamma,model_state , story_len, stored_state =False,popart_update = False): 
    global mu,vt,sigma

    state_mem = []
    if stored_state:
        [models[i].set_state(model_state[i]) for i in range(4)]
        state_mem.append([models[i].get_state() for i in range(4)])
        for i in range(story_len):
            _,_ = models[V](state[i])
            _,_,_,_,_ = models[P](state[i])
            _,_,_,_ = models[Q](state[i],action[i])
            _,_ = models[TV](state[i])

            state_mem.append([models[i].get_state() for i in range(4)])
    

    loss_q1=[]
    loss_q2=[]
    loss_value=[]
    loss_policy=[]
    next_exp_q = []
    with torch.no_grad():
        [models[i].set_state(model_state[i]) for i in range(4)]
        for i in range(n_step):
            _,_ = models[TV](state[i])
        for i in range(story_len):
            tar_v,_ = models[TV](state[i+n_step])
            next_q = reward[i] + ( gamma[i+n_step]**n_step)*tar_v
            next_exp_q.append(next_q)
        next_exp_q  =torch.stack(next_exp_q,0)
    
    if popart_update:
        new_mu = ((1-POPART_BETA)*mu +POPART_BETA*next_exp_q.mean()).detach().to(dev)
        vt = ((1-POPART_BETA)*vt +POPART_BETA*next_exp_q.pow(2).mean()).detach().to(dev)
        new_sigma = torch.sqrt(vt-mu**2).to(dev)
        models[V].last.weight.data = models[V].last.weight.data*sigma/new_sigma
        models[V].last.bias.data = (models[V].last.bias.data*sigma +mu-new_mu)/new_sigma
        models[Q].last1.weight.data = models[Q].last1.weight.data*sigma/new_sigma
        models[Q].last1.bias.data = (models[Q].last1.bias.data*sigma +mu-new_mu)/new_sigma
        models[Q].last2.weight.data = models[Q].last2.weight.data*sigma/new_sigma
        models[Q].last2.bias.data = (models[Q].last2.bias.data*sigma +mu-new_mu)/new_sigma

        mu = new_mu
        sigma = new_sigma



    for i in range(story_len):
        policy_state = models[P].get_state()

        exp_q1,exp_q2,_,_ = models[Q](state[i],action[i])
        new_action, log_prob, z,mean,log_std = models[P](state[i],eval=True)
        exp_v,_ = models[V](state[i])

        next_q = (next_exp_q[i] -mu)/sigma
        l_q1 = (next_q - exp_q1).pow(2)
        l_q2 = (next_q - exp_q2).pow(2)
        loss_q1.append(l_q1)
        loss_q2.append(l_q2)

        models[P].set_state(policy_state)
        new_q1,new_q2,_,_ = models[Q](state[i],new_action)
        new_exp_q = torch.min(new_q1,new_q2)

        next_value = new_exp_q -(ALPHA *log_prob)
        l_value = (next_value - exp_v).pow(2)
        loss_value.append(l_value)

        mean_loss = 0.001*mean.pow(2).mean(1)
        std_loss = 0.001*log_std.pow(2).mean(1)
        l_policy = ((ALPHA*log_prob) - new_exp_q).mean(1) + mean_loss + std_loss
        loss_policy.append(l_policy)
    
    loss_q1 = torch.stack(loss_q1,1)
    loss_q2 = torch.stack(loss_q2,1)
    loss_value = torch.stack(loss_value,1)
    loss_policy = torch.stack(loss_policy,1).reshape(loss_q1.size())

    return loss_q1,loss_q2,loss_value, loss_policy,state_mem

        


class actor_worker():
    def __init__(self,a_id,num_frames,shared_state,shared_queue, eps=0.1,block=True):
        super(actor_worker,self).__init__()
        self.shared_state = shared_state
        self.shared_queue = shared_queue
        self.block = block
        self.num_frames = num_frames
        self.a_id = a_id
        self.eps = eps 
        self.env = env_cover(env_id)
        print(f'#{a_id} start')
        self.win_epsil = vis.line(Y=torch.tensor([0]),opts=dict(title='epsilon'+str(a_id)))
        self.win_r = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'+str(a_id)))
        self.win_exp_q = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_q'+str(a_id)))
        self.win_exp_a = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_a'+str(a_id)))
        self.win_exp_v = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_v'+str(a_id)))
        self.win_exp_aprob = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_aprob'+str(a_id)))

              
        self.policy_net = Actor(s_dim, a_dim, dev ).to(dev)
        self.q_net = QCritic(s_dim, a_dim, dev ).to(dev)
        self.v_net = VCritic(s_dim, a_dim, dev ).to(dev)
        self.rnd_model  = RND(s_dim).to(dev)
        self.policy_net.load_state_dict(self.shared_state["p"].state_dict())
        self.q_net.load_state_dict(self.shared_state["q"].state_dict())
        self.v_net.load_state_dict(self.shared_state["v"].state_dict())

        self.random_prob= torch.distributions.normal.Normal(torch.zeros([a_dim]),torch.ones([a_dim])*0.1)



    def run(self):

        episode_reward=0
        local_mem = []
        done = True
        gamma = 0.997
        state = self.env.reset()
        q_val=[]
        aprob_val=[]
        a_val=[]
        v_val=[]
        qa_val=[]
        act_prob = torch.zeros([1,a_dim])

        for frame_idx in range(self.num_frames):
            if done:
                if len(local_mem)!=0:
                    with self.shared_state["vis"].get_lock():
                        vis.line(X=torch.tensor([frame_idx]), Y=torch.tensor([episode_reward]), win = self.win_r, update='append')
        #                vis.line(X=torch.tensor([frame_idx]), Y=torch.tensor([epsilon]), win = win_epsil, update='append')
                        vis.line(Y=torch.cat(q_val,0), win= self.win_exp_q, opts=dict(title='exp_q'+str(self.a_id)))
                        vis.line(Y=torch.stack(v_val,0).view(-1,1), win= self.win_exp_v, opts=dict(title='exp_v'+str(self.a_id)))
                        vis.line(Y=torch.stack(a_val,0), win= self.win_exp_a, opts=dict(title='exp_a'+str(self.a_id)))
                        vis.line(Y=torch.cat(aprob_val,0), win= self.win_exp_aprob, opts=dict(title='exp_aprob'+str(self.a_id)))
#                        vis.line(Y=torch.cat(q_val,0), win= self.win_exp_qa, opts=dict(title='exp_q'+str(self.a_id)))
                        
                        
                        
                    for i in range(n_step):
                        local_mem.append([torch.zeros(state.size()), torch.zeros(act_prob.size()),0,0,0,0])
                        
        #            for i in range(len(local_mem)-n_step):
        #                local_mem[i][5] = 0.99 if local_mem[i][3]!=0 else 0 
        #                state = local_mem[i][0]
        #                next_state = local_mem[i+n_step][0]
        #                
        ##                state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1,i+1)],1)
        ##                next_state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1+n_step,i+1+n_step)],1)
        #                pred , targ = rnd_model(state.to(dev),next_state.to(dev))
        #                i_reward = ((pred-targ)**2).mean().item()
        #                local_mem[i][4] = i_reward
            
                    for i in range(len(local_mem)-n_step):
                        local_mem[i][2] = sum([local_mem[i+j][2] *(local_mem[i+j][3]**j) for j in range(n_step)])
        #                local_mem[i][4] = sum([local_mem[i+j][4] *(0.99**j) for j in range(n_step)])
            
        #            ll = []
        #            for i in range(len(local_mem)-n_step):
        #                ll.append(local_mem[i][4])
        #            win_ir = vis.line(Y=torch.tensor(ll),win= win_ir)
                    with torch.no_grad():
                        self.policy_net.reset_state()
                        self.q_net.reset_state()
    #                    targetQ.reset_state()
    #                    mhx,mcx= actor.get_state()
    #                    thx,tcx= targetQ.get_state()
                        state,action,reward,gamma,ireward,igamma = zip(*local_mem)
            
                        b_len = len(local_mem)
                        state = torch.stack(state).cpu()
                        action = torch.stack(action).cpu()
                        reward = torch.Tensor(reward).reshape((b_len,1,1)).cpu()
                        gamma = torch.Tensor(gamma).reshape((b_len,1,1)).cpu()
                        ireward = torch.Tensor(ireward).reshape((b_len,1,1)).cpu()
                        igamma = torch.Tensor(igamma).reshape((b_len,1,1)).cpu()
                        
                        blocking = True if self.shared_queue.qsize()>max_shared_q_size and self.block else False
                        self.shared_queue.put([state ,action,reward,gamma,ireward,igamma ],block=blocking)
                        
                        
                    
                    
                    if self.block == False:
                        return 0
            
                state = self.env.reset()
                episode_reward=0
                gamma = 0.997
                local_mem = []
                self.policy_net.reset_state()
                self.q_net.reset_state()
    #            targetQ.reset_state()
                q_val = []
                a_val = []
                v_val = []
                aprob_val = []
                qa_val = []
                
            while True:
                with self.shared_state["wait"].get_lock():
                    if self.shared_state["wait"].value > 0:
                        self.shared_state["wait"].value -=1
                        break
                time.sleep(0.1)
                        
        #    epsilon= 0.01**(EPS_CONST*frame_idx/num_frames)
        #    epsilon= eps
            
            with torch.no_grad():
    #            mhx,mcx = actor.get_state()
    #            thx,tcx = targetQ.get_state()
    #            state_mem.append([mhx,mcx,thx,tcx])
    #            state_mem.append([mhx,mcx])
                action, log_prob, z, mu,log_std  = self.policy_net(state)
                Q1,Q2,_,_ = self.q_net(state,action)
                V,_ = self.v_net(state)

            env_action = action.argmax()
            aprob_val.append(action)
            a_val.append(env_action)
            v_val.append(V.detach())
            q_val.append(torch.cat([Q1,Q2],1).detach())
            #qa_val.append(Q.gather(1,act.view(-1,1)).detach())
    #        if vis_render:
    #            vis.image(state.view(84,84),win = win_img)
                
            next_state , reward, done ,_ = self.env.step(env_action.item())
            local_mem.append([state.cpu(),action.cpu() ,reward, gamma, 0 , 0])
            #self.env.render()
            
            state = next_state
            episode_reward += reward
        
        
            if self.shared_state["update"][self.a_id]:
                self.policy_net.load_state_dict(self.shared_state["p"].state_dict())
                self.q_net.load_state_dict(self.shared_state["q"].state_dict())
                self.v_net.load_state_dict(self.shared_state["v"].state_dict())
                self.shared_state["update"][self.a_id]=False
                
#                print('actor_update',action.value[0].weight[0][0:5].detach())
        
        
        print('done')
        self.env.close()
    

class learner_worker():
    def __init__(self,max_id,num_frames,shared_state,shared_queue,block=True):
        super(learner_worker,self).__init__()
        self.shared_state = shared_state
        self.shared_queue = shared_queue
        self.max_id = max_id
        self.num_frames = num_frames
        self.block = block

        self.win_ir = vis.line(Y=torch.tensor([0]),opts=dict(title='ireward'))
        self.win_q1 = vis.line(Y=torch.tensor([0]),opts=dict(title='Q1_loss'))
        self.win_q2 = vis.line(Y=torch.tensor([0]),opts=dict(title='Q2_loss'))
        self.win_v = vis.line(Y=torch.tensor([0]),opts=dict(title='v_loss'))
        self.win_p = vis.line(Y=torch.tensor([0]),opts=dict(title='policy_loss'))
        self.win_mu = vis.line(Y=torch.tensor([0]),opts=dict(title='popart_mu'))
        self.win_sigma = vis.line(Y=torch.tensor([0]),opts=dict(title='popart_sigma'))
        
        self.policy_net = Actor(s_dim, a_dim, dev ).to(dev)
        self.t_policy_net = Actor(s_dim, a_dim, dev ).to(dev)
        self.q_net = QCritic(s_dim, a_dim, dev ).to(dev)
        self.t_q_net = QCritic(s_dim, a_dim, dev ).to(dev)
        self.v_net = VCritic(s_dim, a_dim, dev ).to(dev)
        self.t_v_net = VCritic(s_dim, a_dim, dev ).to(dev)
        
        
        self.rnd_model  = RND(s_dim).to(dev)
        
        self.policy_net.load_state_dict(self.shared_state["p"].state_dict())
        self.t_policy_net.load_state_dict(self.shared_state["p"].state_dict())
        self.q_net.load_state_dict(self.shared_state["q"].state_dict())
        self.t_q_net.load_state_dict(self.shared_state["q"].state_dict())
        self.v_net.load_state_dict(self.shared_state["v"].state_dict())
        self.t_v_net.load_state_dict(self.shared_state["v"].state_dict())
    
        self.models=[self.v_net, self.policy_net,self.q_net,self.t_v_net,self.t_policy_net,self.t_q_net,self.rnd_model]


        
        self.p_optimizer = optim.Adam(self.policy_net.parameters(),a_lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(),c_lr)
        self.v_optimizer = optim.Adam(self.v_net.parameters(),v_lr)
        self.rnd_optimizer = optim.Adam(self.rnd_model.parameters(),rnd_lr)
        
        
        self.replay_buffer = ReplayBuffer(mem_size,self.models,self.shared_state)
        self.q1_loss = 0
        self.q2_loss = 0
        self.p_loss = 0
        self.v_loss = 0


    def soft_update(self,target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def push_buffer(self,blocking=False):
        if self.shared_queue.qsize()!=0:
#            while shared_queue.qsize() != 0:
            data = self.shared_queue.get(block=blocking)
            self.replay_buffer.push(data)
    
    
    def run(self):


        while len(self.replay_buffer) < start_frame and self.block:
            self.push_buffer(True)
            print(repr(self.replay_buffer),end='\r')
        
        
        for frame_idx in range(self.num_frames):
            print(repr(self.replay_buffer),end='\r')
            self.push_buffer(False)
    
            self.update(frame_idx)
            print('#learner  q1:{:.5f} q2:{:.5f}, v:{:.5f}, p:{:5f}'.format(self.q1_loss,self.q2_loss,self.v_loss,self.p_loss))

            with self.shared_state["vis"].get_lock():
                self.win_q1 = vis.line(X=torch.tensor([frame_idx]),Y=self.q1_loss.detach().view(1,-1),win=self.win_q1,update ='append')
                self.win_q2 = vis.line(X=torch.tensor([frame_idx]),Y=self.q2_loss.detach().view(1,-1),win=self.win_q2,update ='append')
                self.win_v = vis.line(X=torch.tensor([frame_idx]),Y=self.v_loss.detach().view(1,-1),win=self.win_v,update ='append')
                self.win_p= vis.line(X=torch.tensor([frame_idx]),Y=self.p_loss.detach().view(1,-1),win=self.win_p,update ='append')
            
            with self.shared_state["wait"].get_lock():
                self.shared_state["wait"].value +=10
            
                
                
#                update_target(targetQ,mainQ)
            if frame_idx % 3 == 0:
    #        if random.random() < 1/20 :
                self.shared_state["p"].load_state_dict(self.models[P].state_dict())
                self.shared_state["q"].load_state_dict(self.models[Q].state_dict())
                self.shared_state["v"].load_state_dict(self.models[V].state_dict())
                for i in range(self.max_id):
                    self.shared_state["update"][i]=True
            if self.block == False:
                return 0
    def update(self,frame_idx):
        global mu,sigma
        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,hxcx, burn_state, burn_action = self.replay_buffer.sample(batch_size)
        burned_state = []
#        [models[i].reset_state() for i in range(4)]
#        model_state = [self.models[i].get_state() for i in range(4)]
        
        
        with torch.no_grad():
            for i in range(batch_size):
                [self.models[j].reset_state() for j in range(4)]
                
                [self.models[j].set_state(hxcx[i][2*j:2*j+2]) for j in range(4)]
                        
                
                for j in range(len(burn_state[i])):
                    _ = self.models[V](burn_state[i][j])
                    _,_,_,_,_ = self.models[P](burn_state[i][j])
                    _,_,_,_ = self.models[Q](burn_state[i][j],burn_action[i][j])
                    _ = self.models[TV](burn_state[i][j])

                model_state = [self.models[i].get_state() for i in range(4)]
                burned_state.append( torch.stack(model_state,0) )
                
            
        burned_state = torch.cat (burned_state,2)
        
        loss_q1,loss_q2,loss_v,loss_p,_ = calc_td(self.models,state,action,reward,gamma,ireward,igamma,burned_state,seq_len,popart_update=True)
        self.win_mu = vis.line(X=torch.tensor([frame_idx]),Y=mu.view(1,-1),win=self.win_mu,update='append')
        self.win_sigma = vis.line(X=torch.tensor([frame_idx]),Y=sigma.view(1,-1),win=self.win_mu,update='append')

        self.q_optimizer.zero_grad()
        self.q1_loss = loss_q1.mean()
        self.q1_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        self.q_optimizer.zero_grad()
        self.q2_loss = loss_q2.mean()
        self.q2_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        self.v_optimizer.zero_grad()
        self.v_loss = loss_v.mean()
        self.v_loss.backward(retain_graph=True)
        self.v_optimizer.step()
        
        self.p_optimizer.zero_grad()
        self.p_loss = loss_p.mean()
        self.p_loss.backward(retain_graph=True)
        self.p_optimizer.step()

        self.soft_update(self.models[TP],self.models[P],SOFT_TAU)
        self.soft_update(self.models[TQ],self.models[Q],SOFT_TAU)
        self.soft_update(self.models[TV],self.models[V],SOFT_TAU)
        
        with torch.no_grad():
            td_loss = (loss_q1.sqrt() + loss_q2.sqrt() + loss_v.sqrt())
            for i in range(len(epi_idx)):
                self.replay_buffer.priority_update(epi_idx[i],seq_idx[i],td_loss[i])
            self.replay_buffer.mem_remove(epi_idx)

        return 0,0

        
#    #    if len(replay_buffer)==0:
#        if block==False:
#            if shared_queue.qsize()<2 :
#                print('return  shared q size > 2 ')
#                return 0
#            data = shared_queue.get(block=True)
#            replay_buffer.push(data)
#        

    

def act_process(idd,num_frames,shared_state,shared_queue,eps):
    act=  actor_worker(idd,num_frames,shared_state,shared_queue,eps,True)
    act.run()        
def lea_preocess(idd,num_frames,shared_state,shared_queue):
    lea = learner_worker(idd,num_frames,shared_state,shared_queue,True)
    lea.run()




if __name__ == '__main__':
    os.system('cls')
    
    vis.close()
      
    num_processes = 1
        
    shared_queue = mp.Queue()
    shared_state = dict()
    
    shared_state["p"] = Actor(s_dim, a_dim, dev ).share_memory()
    shared_state["q"] = QCritic(s_dim, a_dim, dev ).share_memory()
    shared_state["v"] = VCritic(s_dim, a_dim, dev ).share_memory()
    
    shared_state["update"] = mp.Array('i', [0 for i in range(num_processes)])
#    shared_state["wait"] = mp.Array('i', [0 for i in range(num_processes)])
    shared_state["vis"] = mp.Value('i',0)
    shared_state["wait"] = mp.Value('i',0)
    shared_state["wait"].value = start_frame*10
    
    
    act = actor_worker(0,num_frames,shared_state,shared_queue,0.1,False)
    act.run()
    act.run()
    act.run()
    lea = learner_worker(1,num_frames,shared_state,shared_queue,False)
    lea.push_buffer()
    lea.push_buffer()
    lea.push_buffer()
    
#    for i in range(100):
#        act.run()
#        lea.run()
    
#    for i in range(100):
#        actor_process(0,num_frames,shared_state,shared_queue,False)
#        actor_process(0,num_frames,shared_state,shared_queue,False)
#        learner_process(1,num_frames,shared_state,shared_queue,False)
#    time.sleep(10)
###    
    proc_list = []
    proc_list.append(mp.Process(target=lea_preocess, args=(num_processes,num_frames,shared_state,shared_queue)))
    eps = [0.1,0.2,0.4,0.3,0.2,0.6,0.4,0.6,0.2,0.4]
    for i in range(num_processes):
        proc_list.append( mp.Process(target=act_process, args=(i,num_frames,shared_state,shared_queue,eps[i])) )


    for proc in proc_list:
        proc.start()
        
    try:
        for proc in proc_list:
            proc.join()
    except:
        print('qclose')
        shared_queue.close()
        print('process close')
        for proc in proc_list:
            proc.terminate()
            
            
        shared_queue.join_thread()
    
#    
