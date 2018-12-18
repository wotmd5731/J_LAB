
import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
vis = visdom.Visdom(port = 8097)
vis.close()

win_r = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'))
win_ir = vis.line(Y=torch.tensor([0]),opts=dict(title='ireward'))
win_l0 = vis.line(Y=torch.tensor([0]),opts=dict(title='loss'))
win_l1 = vis.line(Y=torch.tensor([0]),opts=dict(title='rnd_loss'))
#win_4 = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'))

from time import time
ttime= time()
def time_check(num=0):
    global ttime
    print(f'{num} time:{time()-ttime}')
    ttime = time()

"""
+double
+dueling
+episodic mem
+nstep
+per
+image version
+frame_stack
+RND

"""
frame_stack = 4
n_step = 5
PER_alpha = 0.9  # 0 is uniform per
count_episode = False
RND_const = 0
start_frame = 1000
num_frames = 50000
batch_size =128


env_id = 'CartPole-v0'
#env_id = 'MountainCar-v0'
env = gym.make(env_id)
s_dim = 1*frame_stack
a_dim = 2


use_cuda = False
use_cuda = torch.cuda.is_available()
dev = torch.device('cuda' if use_cuda else 'cpu')
print(dev)

import torch.utils.data

from collections import deque

class ReplayBuffer():
    def __init__(self,capacity):
        self.count = 0
        self.capacity = capacity
        self.buffer = deque(maxlen= capacity)
    def push(self, data, td_loss):
        self.count += len(data) if not count_episode else 1
        td_loss_total = sum(td_loss)/len(td_loss)
        self.buffer.append([data,td_loss,td_loss_total])
        while self.count > self.capacity:
            self.count -= len(self.buffer.popleft()[0]) if not count_episode else 1

    def sample(self,batch_size):
        weight = [self.buffer[i][2] for i in range(len(self.buffer))]
        batch_epi = list(torch.utils.data.WeightedRandomSampler(torch.stack(weight),batch_size, True))
        s = []
        for episode_idx in batch_epi:
            episode = self.buffer[episode_idx][0]
            td_loss = self.buffer[episode_idx][1]
            ii = list(torch.utils.data.WeightedRandomSampler(td_loss , 1, True))[0]
            state = torch.cat([episode[j if j>=0 else  0][0] for j in range(ii-frame_stack+1,ii+1)],1)
            _, action , reward , _, ireward, _ = episode[ii]
            nstate = torch.cat([episode[j if j>=0 else  0][0] for j in range(ii-frame_stack+1+n_step,ii+1+n_step)],1)
            _,_,_, gamma,_,igamma = episode[ii+n_step]
            s.append([episode_idx,ii,state,action,reward,gamma,ireward,igamma,nstate])

        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,nstate = zip(*s)
        shape = (batch_size,-1)
        state   = torch.cat(state,0).reshape((batch_size, s_dim, 84,84)).to(dev)
        nstate   = torch.cat(nstate,0).reshape((batch_size, s_dim, 84,84)).to(dev)
        action = torch.LongTensor(action).reshape(shape).to(dev)
        epi_idx = torch.LongTensor(epi_idx).reshape(shape).to(dev)
        seq_idx = torch.LongTensor(seq_idx).reshape(shape).to(dev)
        reward = torch.Tensor(reward).reshape(shape).to(dev)
        gamma = torch.Tensor(gamma).reshape(shape).to(dev)
        ireward = torch.Tensor(ireward).reshape(shape).to(dev)
        igamma = torch.Tensor(igamma).reshape(shape).to(dev)
        
        return epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,nstate
    def priority_update(self,epi_idx,seq_idx,new_td_loss):
        self.buffer[epi_idx][1][seq_idx]=new_td_loss
        self.buffer[epi_idx][2] = sum(self.buffer[epi_idx][1][:])/len(self.buffer[epi_idx][1][:])
    def __len__(self):
        return self.count
    def __repr__(self):
        return '\rmem size: {}/{} ' .format(self.count, self.capacity)



class Flatten(nn.Module):
    def forward(self,inputs):
        return inputs.view(inputs.size(0),-1)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs ):
        super(DQN,self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(num_inputs,64,8,stride= 4),
                nn.ReLU(),
                nn.Conv2d(64,64,4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,3,stride=1),
                nn.ReLU(),
                Flatten()
                )
        size=7*7*64
        self.advantage = nn.Sequential(
                nn.Linear(size,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,num_outputs),
                )
        self.value = nn.Sequential(
                nn.Linear(size,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,1),
                )
        self.iadvantage = nn.Sequential(
                nn.Linear(size,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,num_outputs),
                )
        self.ivalue = nn.Sequential(
                nn.Linear(size,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,1),
                )

    def forward(self,x):
        x = self.feature(x)
        adv = self.advantage(x)
        val = self.value(x)
        iadv = self.iadvantage(x)
        ival = self.ivalue(x)
        
        Q = val + adv - adv.mean()
        iQ = ival + iadv - iadv.mean()
        Qa = Q.argmax(1).view(-1,1)
        iQa = iQ.argmax(1).view(-1,1)
        return Q,Qa,iQ,iQa

class RND(nn.Module):
    def __init__(self,num_inputs):
        super(RND,self).__init__()
        self.target= nn.Sequential(
                nn.Conv2d(num_inputs*2,64,8,stride=4),
                nn.ReLU(),
                nn.Conv2d(64,64,4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,3,stride=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(64*7*7,256),
                nn.ReLU(),
                nn.Linear(256,256),
                )
        self.predictor = nn.Sequential(
                nn.Conv2d(num_inputs*2,64,8,stride=4),
                nn.ReLU(),
                nn.Conv2d(64,64,4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,3,stride=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(64*7*7,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,256),
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



main_model = DQN(s_dim, a_dim ).to(dev)
target_model = DQN(s_dim, a_dim ).to(dev)
rnd_model  = RND(s_dim).to(dev)

optimizer = optim.Adam(main_model.parameters(),0.0006)
rnd_optimizer = optim.Adam(rnd_model.parameters(),0.0001)

replay_buffer = ReplayBuffer(10000)

def update_target(cur, tar):
    tar.load_state_dict(cur.state_dict())

update_target(main_model,target_model)

def compute_td_loss(epi_idx,seq_idx,state,action,reward,gamma,ireward,igamma,next_state):
    q_v,_, iq_v, _ = main_model(state)
    nq_v, nq_a, niq_v, niq_a = main_model(next_state)
    ntq_v , _, ntiq_v,_ = target_model(next_state)

    q_v = q_v.gather(1,action)
    iq_v = iq_v.gather(1,action)
    
    nqv = ntq_v.gather(1,nq_a)
    niqv = ntiq_v.gather(1,niq_a)
    
    exp_qv = reward + (gamma**n_step) * nqv
    exp_iqv = ireward + (igamma**n_step) * niqv

    td_loss = (q_v - exp_qv.detach()).pow(2)
    itd_loss = (iq_v - exp_iqv.detach()).pow(2)
    return td_loss,itd_loss


import torchvision
togray = torchvision.transforms.Grayscale()
toten = torchvision.transforms.ToTensor()
resize = torchvision.transforms.Resize((84,84))
topil = torchvision.transforms.ToPILImage()
def obs_preproc(x):
    xten = toten(togray(resize(topil(x))))
    return xten.view(1,1,84,84)



all_rewards = []
state = env.reset()
state = obs_preproc(env.render(mode='rgb_array')).to(dev)
vis_render=True
episode_reward=0
if vis_render:
    win_img = vis.image(state.view(84,84))

local_mem = [[state,0,0,0,0,0]]


for frame_idx in range(num_frames):
    print(repr(replay_buffer),end='\r')
    epsilon= 0.01**(frame_idx/num_frames)
    
    mem_len = len(local_mem)
    stack_state = [local_mem[j if j>=0 else 0][0] for j in range(mem_len - frame_stack+1,mem_len)]
    stack_state.append(state)
    stack_state = torch.cat(stack_state,1)

    if random.random() > epsilon:
        q,qa,iq,iqa = main_model(stack_state)
        action = qa.item()
    else:
        action = random.randrange(a_dim)

    if vis_render:
        win_img = vis.image(state.view(84,84),win = win_img)
    next_state , reward, done ,_ = env.step(action)
    gamma = 0.887 if not done else 0
    next_state = obs_preproc(env.render(mode='rgb_array')).to(dev)

    local_mem.append([state, action ,reward, gamma, 0 , 0])
    state = next_state
    episode_reward += reward

    if done:
        win_r = vis.line(X=torch.tensor([frame_idx]), Y=torch.tensor([episode_reward]), win = win_r, update='append')
        for i in range(n_step):
            local_mem.append([torch.zeros(state.size()).to(dev),0,0,0,0,0])
        for i in range(len(local_mem)-n_step):
            local_mem[i][5] = 0.99 if local_mem[i][3]!=0 else 0 
            

            state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1,i+1)],1)
            next_state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1+n_step,i+1+n_step)],1)
            pred , targ = rnd_model(state.to(dev),next_state.to(dev))
            i_reward = ((pred-targ)**2).mean().item()
            local_mem[i][4] = i_reward

        for i in range(len(local_mem)-n_step):
            local_mem[i][2] = sum([local_mem[i+j][2] *(0.997**j) for j in range(n_step)])
            local_mem[i][4] = sum([local_mem[i+j][4] *(0.99**j) for j in range(n_step)])

        ll = []
        for i in range(len(local_mem)-n_step):
            ll.append(local_mem[i][4])
        win_ir = vis.line(Y=torch.tensor(ll),win= win_ir)

        def calc_priority():
            priority = []
            with torch.no_grad():
                for i in range(len(local_mem)-n_step):
                    shape = (1,-1)
                    #state = local_mem[i][0]
                    #next_state = local_mem[i+n_step][0]
                    action = [local_mem[i][1]]
                    reward  = [local_mem[i][2]]
                    gamma = [local_mem[i+n_step][3]]
                    ireward = [local_mem[i][4]]
                    igamma = [local_mem[i+n_step][5]]
                    
                    state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1,i+1)],1)
                    next_state = torch.cat([local_mem[j if j>=0 else  0][0] for j in range(i-frame_stack+1+n_step,i+1+n_step)],1)
                    
                    action = torch.LongTensor(action).reshape(shape).to(dev)
                    reward = torch.Tensor(reward).reshape(shape).to(dev)
                    gamma = torch.Tensor(gamma).reshape(shape).to(dev)
                    ireward = torch.Tensor(ireward).reshape(shape).to(dev)
                    igamma = torch.Tensor(igamma).reshape(shape).to(dev)
        
                    td_l, itd_l = compute_td_loss(0,0,state,action,reward,gamma,ireward,igamma,next_state)
                    priority.append((td_l-itd_l).abs().view(-1)**PER_alpha)
                priority = torch.cat(priority)
                return priority

        priority = calc_priority()
        replay_buffer.push(local_mem,priority)
        episode_reward=0
        all_rewards = []
        state = env.reset()
        state = obs_preproc(env.render(mode='rgb_array')).to(dev)
        local_mem = [[state,0,0,0,0,0]]
                    
    if len(replay_buffer) > start_frame and frame_idx%10 == 0:
        def update():
            epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,nstate = replay_buffer.sample(batch_size)
            td, itd = compute_td_loss(epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,nstate)
    
            loss = (td+itd).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            pm,tm = rnd_model(state,nstate)
            rnd_loss = ((pm-tm)**2).mean()
            rnd_optimizer.zero_grad()
            rnd_loss.backward()
            rnd_optimizer.step()
    
            for i in range(len(epi_idx)):
                replay_buffer.priority_update(epi_idx[i],seq_idx[i],(td[i]+itd[i]).abs().detach().view(-1)**PER_alpha)
            return loss.item(),rnd_loss.item()
        
        loss, rnd_loss = update()
        win_l0 = vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([loss]),win=win_l0,update ='append')
        win_l1 = vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([rnd_loss]),win=win_l1,update ='append')


    if frame_idx%100 == 0:
        update_target(main_model,target_model)















