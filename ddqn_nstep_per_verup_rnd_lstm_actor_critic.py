
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
max_shared_q_size = 5
frame_stack = 1
n_step = 5
PER_alpha = 0.9  # 0 is uniform per
count_episode = False
RND_const = 0
start_frame = 1000
num_frames = 50000
batch_size =32
vis_render=True
EPS_CONST = 1
a_lr = 0.00006
c_lr = 0.0006
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
        #ss = np.delete(ss,[1,3])
        return torch.from_numpy(ss).float().view(1,s_dim).to(dev)
    #return obs_preproc(env.render(mode='rgb_array')).to(dev)
    def step(self,act):
        ss,rr,dd,_ = self.env.step(act)
        #ss = np.delete(ss,[1,3])
        return torch.from_numpy(ss).float().view(1,s_dim).to(dev),rr,dd,0

    def close(self):
        self.env.close()


cnn_enable = False
vis_render=False
#s_dim = 2
s_dim = 4
state_shape = (1,1,s_dim)
#a_dim = 3

#env_id = 'MountainCar-v0'



env = env_cover(env_id)


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
            
            td_loss, state_mem = calc_td(self.models,state,action,reward,gamma,ireward,igamma,
                               model_state, 
                               b_len-n_step, stored_state=True)
        
        
        
        
        self.count += data[0].size(0) if not count_episode else 1
        priority = []
        eta = 0.9
        td_loss = td_loss.view(-1)
        for i in range(len(td_loss)-seq_len):
            p = (eta*td_loss[i:i+seq_len].max()+(1.-eta)*td_loss[i:i+seq_len].mean())**PER_alpha
            priority.append(p)
            
        priority = torch.stack(priority).view(-1)
#        td_loss_total = sum(priority)/len(priority)
        td_loss_total = priority.max()
        with self.shared_state["vis"].get_lock():
            vis.bar(X=td_loss.cpu().view(-1,1), win= self.win_bar_td, opts=dict(title='push td_loss'))
        self.buffer.append([data,td_loss,priority,td_loss_total,state_mem])
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
            brun_state = episode[0][start:ii].to(dev)
            
            model_state = torch.cat(state_mem[start],0)
            
#            model_state = torch.cat([ahxcx,tahxcx,chxcx,tchxcx],0)
            
            state   =episode[0][ii:ii+seq_len+n_step]  
            action  =episode[1][ii:ii+seq_len+n_step]
            reward  =episode[2][ii:ii+seq_len+n_step]
            gamma   =episode[3][ii:ii+seq_len+n_step] 
            ireward =episode[4][ii:ii+seq_len+n_step]
            igamma  =episode[5][ii:ii+seq_len+n_step]
            
            s.append([episode_idx,ii,state,action,reward,gamma,ireward,igamma, model_state ,brun_state])

        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,model_state, burn_state = zip(*s)
        
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
        
        return epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,hxcx, burn_state
    
    def priority_update(self,epi_idx,seq_idx,loss):
        td_array = self.buffer[epi_idx][1]
#        priority = self.buffer[epi_idx][2]
#        total_priority = self.buffer[epi_idx][3]
        
        for i in range(seq_len):
            td_array[seq_idx+i] = loss[i].abs()
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

class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs,  dev ):
        super(Critic,self).__init__()
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
        
        self.advantage = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,num_outputs),
                )
        self.value = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.iadvantage = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,num_outputs),
                )
        self.ivalue = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,1),
                )
        self.hx = None
        self.cx = None

        self.dev = dev
        
    def forward(self,x):
#        aa = torch.ones(x.size())*a
#        x = torch.cat([x,aa],dim=1)
        x = self.feature(x)
        
        if self.hx is None: 
            self.hx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            self.cx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            
        self.hx, self.cx = self.lstm(x , (self.hx, self.cx))
        
        x= self.hx
        
        adv = self.advantage(x)
        val = self.value(x)
        iadv = self.iadvantage(x)
        ival = self.ivalue(x)
        
        Q = val + adv - adv.mean()
        iQ = ival + iadv - iadv.mean()
        Qa = Q.argmax(1).view(-1,1)
        iQa = iQ.argmax(1).view(-1,1)
        return Q,Qa,iQ,iQa

    
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
        
        self.net = nn.Sequential(
                nn.Linear(self.lstm_size,128),nn.PReLU(),
                nn.Linear(128,128),nn.PReLU(),
                nn.Linear(128,num_outputs),nn.Tanh(),
                )

        self.hx = None
        self.cx = None

        self.dev = dev
        
    def forward(self,x):
        
        x = self.feature(x)
        
        if self.hx is None: 
            self.hx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            self.cx = torch.zeros((x.size(0) ,self.lstm_size)).to(self.dev)
            
        self.hx, self.cx = self.lstm(x , (self.hx, self.cx))
        
        x= self.hx
        
        act_prob = self.net(x)
        
        m = Categorical(act_prob)
        act = m.sample()
        
        return  act,act_prob

    
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


def calc_td(models,state, action, reward,gamma,ireward,igamma,model_state , story_len, stored_state =False): 
    y_t_hat = []
    iy_t_hat = []
    state_mem = []
    losses = []
#    with torch.no_grad():
    
    
    
    if stored_state:
        models[0].set_state(model_state[0])
        models[1].set_state(model_state[1])
        models[2].set_state(model_state[2])
        models[3].set_state(model_state[3])
        state_mem.append([models[i].get_state() for i in range(4)])
        for i in range(story_len):
            _,_ = models[0](state[i])
            _,_ = models[1](state[i])
            _,_,_,_ = models[2](state[i])
            _,_,_,_ = models[3](state[i])
            state_mem.append([models[i].get_state() for i in range(4)])
    

    models[0].set_state(model_state[0])
    models[1].set_state(model_state[1])
    models[2].set_state(model_state[2])
    models[3].set_state(model_state[3])
        
    for i in range(n_step):
        _,_ = models[1](state[i])
        _,_,_,_ = models[3](state[i])
        
        
#        if stored_state:
#            state_mem.append([models[i].get_state() for i in range(4)])
            
    for i in range(story_len):
        act,_ = models[0](state[i])
        tact,_ = models[1](state[i+n_step])
        qv,_,iqv,_ = models[2](state[i])
        tqv,_,tiqv,_ = models[3](state[i+n_step])
        y_t_hat = reward[i] + (gamma[i+n_step]**n_step)*tqv.gather(1,tact.view(-1,1))
        losses.append ( qv.gather(1,action[i]) - y_t_hat.detach())
        
        
#        if stored_state:
#            state_mem.append([models[i].get_state() for i in range(4)])
                
            
#            y_t_hat.append()
#            iy_t_hat.append(ireward[i] + (igamma[i+n_step]**n_step)*iqv.gather(1,tact.view(-1,1)))
    
#    losses=[]
#    
#    [models[i].reset_state() for i in range(4)]
#    [models[i].set_state(model_state[i]) for i in range(4)]
#    
#    for i in range(story_len):
#        q,_,iq,_ = main_model(state[i])
#        td = q.gather(1,action[i]) - y_t_hat[i]
#        itd = iq.gather(1,action[i]) - iy_t_hat[i]
#        losses.append(td+itd)
    
    return torch.cat(losses,1).abs(), state_mem
        




def actor_process(a_id,num_frames,shared_state,shared_queue,block=True, eps=0.1):
    print(f'#{a_id} start')
    win_epsil = vis.line(Y=torch.tensor([0]),opts=dict(title='epsilon'+str(a_id)))
    win_r = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'+str(a_id)))
    win_exp_q = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_q'+str(a_id)))

          
    actor = Actor(s_dim, a_dim, dev ).to(dev)
    critic = Critic(s_dim, a_dim, dev ).to(dev)
    rnd_model  = RND(s_dim).to(dev)
    actor.load_state_dict(shared_state["actor"].state_dict())
    critic.load_state_dict(shared_state["critic"].state_dict())
     
    episode_reward=0
    local_mem = []
    epsilon = 1
    done = True
    gamma = 0.997
    state = env.reset()
    q_val=[]
    a_val=[]
    qa_val=[]
    for frame_idx in range(num_frames):
        if done:
            if len(local_mem)!=0:
                with shared_state["vis"].get_lock():
                    vis.line(X=torch.tensor([frame_idx]), Y=torch.tensor([episode_reward]), win = win_r, update='append')
    #                vis.line(X=torch.tensor([frame_idx]), Y=torch.tensor([epsilon]), win = win_epsil, update='append')
                    vis.line(Y=torch.cat(q_val,0), win= win_exp_q, opts=dict(title='exp_q'+str(a_id)))            
                for i in range(n_step):
                    local_mem.append([torch.zeros(state.size()).to(dev),0,0,0,0,0])
                    
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
                    actor.reset_state()
                    critic.reset_state()
#                    targetQ.reset_state()
#                    mhx,mcx= actor.get_state()
#                    thx,tcx= targetQ.get_state()
                    state,action,reward,gamma,ireward,igamma = zip(*local_mem)
        
                    b_len = len(local_mem)
                    state = torch.stack(state)
                    action = torch.LongTensor(action).reshape((b_len,1,1))
                    reward = torch.Tensor(reward).reshape((b_len,1,1))
                    gamma = torch.Tensor(gamma).reshape((b_len,1,1))
                    ireward = torch.Tensor(ireward).reshape((b_len,1,1))
                    igamma = torch.Tensor(igamma).reshape((b_len,1,1))
                    
                    blocking = True if shared_queue.qsize()>max_shared_q_size and block else False
                    shared_queue.put([state.cpu() ,action,reward,gamma,ireward,igamma ],block=blocking)
                    
                    
                
                
                if block == False:
                    return 0
        
            state = env.reset()
            episode_reward=0
            gamma = 0.997
            local_mem = []
            actor.reset_state()
            critic.reset_state()
#            targetQ.reset_state()
            q_val = []
            a_val = []
            qa_val = []
            
        while True:
            with shared_state["wait"].get_lock():
                if shared_state["wait"].value > 0:
                    shared_state["wait"].value -=1
                    break
            time.sleep(0.01)
                    
    #    epsilon= 0.01**(EPS_CONST*frame_idx/num_frames)
        epsilon= eps
        
        with torch.no_grad():
#            mhx,mcx = actor.get_state()
#            thx,tcx = targetQ.get_state()
#            state_mem.append([mhx,mcx,thx,tcx])
#            state_mem.append([mhx,mcx])
            act, act_prob = actor(state)
            Q,_,_,_ = critic(state)
#            _,_,_,_ = targetQ(state)
            
        action = act.item() if random.random() > epsilon else random.randrange(a_dim)
        a_val.append(act_prob.detach())
        
        q_val.append(Q.detach())
        qa_val.append(Q.gather(1,act.view(-1,1)).detach())
#        if vis_render:
#            vis.image(state.view(84,84),win = win_img)
            
        next_state , reward, done ,_ = env.step(action)
        local_mem.append([state, action ,reward, gamma, 0 , 0])
        
        state = next_state
        episode_reward += reward
    
    
        if shared_state["update"][a_id]:
            actor.load_state_dict(shared_state["actor"].state_dict())
            critic.load_state_dict(shared_state["critic"].state_dict())
            shared_state["update"][a_id]=False
            
            print('actor_update',action.value[0].weight[0][0:5].detach())
    
    
    print('done')
    env.close()
    

class learner_worker(mp.Process):
    def __init__(self,max_id,num_frames,shared_state,shared_queue,block=True):
        super(learner_worker,self).__init__()
        self.shared_state = shared_state
        self.shared_queue = shared_queue
        self.win_ir = vis.line(Y=torch.tensor([0]),opts=dict(title='ireward'))
        self.win_l0 = vis.line(Y=torch.tensor([0]),opts=dict(title='loss'))
        self.win_l1 = vis.line(Y=torch.tensor([0]),opts=dict(title='rnd_loss'))
        
        self.actor = Actor(s_dim, a_dim, dev ).to(dev)
        self.Tactor = Actor(s_dim, a_dim, dev ).to(dev)
        self.critic = Critic(s_dim, a_dim, dev ).to(dev)
        self.Tcritic = Critic(s_dim, a_dim, dev ).to(dev)
        
        
        self.rnd_model  = RND(s_dim).to(dev)
        
        self.actor.load_state_dict(shared_state["actor"].state_dict())
        self.Tactor.load_state_dict(shared_state["actor"].state_dict())
        self.critic.load_state_dict(shared_state["critic"].state_dict())
        self.Tcritic.load_state_dict(shared_state["critic"].state_dict())
    
        self.models=[actor,Tactor, critic,Tcritic]
        
        self.Act_optimizer = optim.Adam(actor.parameters(),a_lr)
        self.Cri_optimizer = optim.Adam(critic.parameters(),c_lr)
        self.rnd_optimizer = optim.Adam(rnd_model.parameters(),rnd_lr)
        
        
        self.replay_buffer = ReplayBuffer(mem_size,models,shared_state)


    def soft_update(target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def run(self):


        while len(self.replay_buffer) < self.start_frame and self.block:
            
            data = self.shared_queue.get(block=True)
            self.replay_buffer.push(data)
            print(repr(self.replay_buffer),end='\r')
        
        
        for frame_idx in range(self.num_frames):
            print(repr(self.replay_buffer),end='\r')
            if self.shared_queue.qsize()!=0:
#            while shared_queue.qsize() != 0:
                data = self.shared_queue.get()
                self.replay_buffer.push(data)
    
            loss, a_loss = self.update()
            print(f'#learner  l:{loss:.5f}')
            with self.shared_state["vis"].get_lock():
                vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([loss]),win=self.win_l0,update ='append')
                vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([a_loss]),win=self.win_l1,update ='append')
            
            with self.shared_state["wait"].get_lock():
                self.shared_state["wait"].value +=3
            
                
            if frame_idx % 4 == 0:
    #        if random.random() < 1/10 :
                self.soft_update(self.models[1],self.models[0],0.3)
                self.soft_update(self.models[3],self.models[2],0.3)
                
#                update_target(targetQ,mainQ)
            if frame_idx % 3 == 0:
    #        if random.random() < 1/20 :
                self.shared_state["actor"].load_state_dict(self.models[0].state_dict())
                self.shared_state["critic"].load_state_dict(self.models[2].state_dict())
                for i in range(self.max_id):
                    self.shared_state["update"][i]=True
            if self.block == False:
                return 0
    def update(self):
            epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,hxcx, burn_state = self.replay_buffer.sample(batch_size)
            aloss = []
            burned_state = []
#            [models[i].reset_state() for i in range(4)]
#            model_state = [self.models[i].get_state() for i in range(4)]
            
            
            with torch.no_grad():
                for i in range(self.batch_size):
                    [self.models[i].reset_state() for i in range(4)]
                    
                    [self.models[j].set_state(hxcx[i][2*j:2*j+2]) for j in range(4)]
                            
                    
                    for j in range(len(burn_state[i])):
                        _,_ = self.models[0](burn_state[i][j])
                        _,_ = self.models[1](burn_state[i][j])
                        _,_,_,_ = self.models[2](burn_state[i][j])
                        _,_,_,_ = self.models[3](burn_state[i][j])
                    model_state = [self.models[i].get_state() for i in range(4)]
                    burned_state.append( torch.stack(model_state,0) )
                    
                
            burned_state = torch.cat (burned_state,2)
            
            loss,_ = self.calc_td(self.models,state, action, reward,gamma,ireward,igamma,burned_state,seq_len) 
            self.Cri_optimizer.zero_grad()
            loss.pow(2).mean().backward()
            self.Cri_optimizer.step()
            
            
            
            
            
            
            [self.models[i].reset_state() for i in range(4)]
            self.models[0].set_state(burned_state[0])
            self.models[1].set_state(burned_state[1])
            self.models[2].set_state(burned_state[2])
            self.models[3].set_state(burned_state[3])
        
            
            for i in range(self.seq_len):
                act,_ = self.models[0](state[i])
                qv,_,iqv,_ = self.models[2](state[i])
                aloss.append( qv.gather(1,act.view(-1,1)) )
            aloss = torch.cat(aloss,1)
            self.Act_optimizer.zero_grad()
            aloss.mean().backward()
            self.Act_optimizer.step()
            

            
        #            pm,tm = rnd_model(state,nstate)
        #            rnd_loss = ((pm-tm)**2).mean()
        #            rnd_optimizer.zero_grad()
        #            rnd_loss.backward()
        #            rnd_optimizer.step()
            
            for i in range(len(epi_idx)):
                self.replay_buffer.priority_update(epi_idx[i],seq_idx[i],loss[i].detach())
            
            return loss.pow(2).mean().item(),aloss.mean()
        
#    #    if len(replay_buffer)==0:
#        if block==False:
#            if shared_queue.qsize()<2 :
#                print('return  shared q size > 2 ')
#                return 0
#            data = shared_queue.get(block=True)
#            replay_buffer.push(data)
#        

    
        


if __name__ == '__main__':
    os.system('cls')
    
    vis.close()
      
    num_processes = 2
    
    shared_queue = mp.Queue()
    shared_state = dict()
    
    shared_state["actor"] = Actor(s_dim, a_dim, dev ).share_memory()
    shared_state["critic"] = Critic(s_dim, a_dim, dev ).share_memory()
    
    shared_state["update"] = mp.Array('i', [0 for i in range(num_processes)])
#    shared_state["wait"] = mp.Array('i', [0 for i in range(num_processes)])
    shared_state["vis"] = mp.Value('i',0)
    shared_state["wait"] = mp.Value('i',0)
    shared_state["wait"].value = start_frame*10
    
    
#    for i in range(100):
#        actor_process(0,num_frames,shared_state,shared_queue,False)
#        actor_process(0,num_frames,shared_state,shared_queue,False)
#        learner_process(1,num_frames,shared_state,shared_queue,False)
#    time.sleep(10)
##    
    proc_list = []
    proc_list.append(mp.Process(target=learner_process, args=(num_processes,num_frames,shared_state,shared_queue)))
    eps = [0.1,0.2,0.4,0.3,0.2,0.6,0.4,0.6,0.2,0.4]
    for i in range(num_processes):
        proc_list.append( mp.Process(target=actor_process, args=(i,num_frames,shared_state,shared_queue,eps[i])) )


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
    
    
