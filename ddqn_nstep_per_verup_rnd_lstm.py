
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
+lstm
"""
max_shared_q_size = 5
frame_stack = 1
n_step = 5
PER_alpha = 0.9  # 0 is uniform per
count_episode = False
RND_const = 0
start_frame = 100
num_frames = 50000
batch_size =128
vis_render=True
EPS_CONST = 1
lr = 0.001
rnd_lr = 0.00001
burn_in_len = 5
mem_size = 20000
seq_len = 7
env_id = 'CartPole-v0'
#env_id = 'MountainCar-v0'
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
s_dim = 4
state_shape = (1,1,s_dim)

env = env_cover(env_id)


use_cuda = False
use_cuda = torch.cuda.is_available()
dev = torch.device('cuda' if use_cuda else 'cpu')
print(dev)

import torch.utils.data

from collections import deque

class ReplayBuffer():
    def __init__(self,capacity):
        self.win_bar = vis.bar(X=torch.rand([10]))
        self.win_bar_td = vis.bar(X=torch.rand([10]))

        self.count = 0
        self.capacity = capacity
        self.buffer = deque(maxlen= capacity)
    def push(self, data, td_loss, state_mem):
        self.count += data[0].size(0) if not count_episode else 1
        priority = []
        eta = 0.9
        td_loss = td_loss.view(-1)
        for i in range(len(td_loss)-seq_len):
            p = (eta*td_loss[i:i+seq_len].max()+(1.-eta)*td_loss[i:i+seq_len].mean())**PER_alpha
            priority.append(p)
            
        priority = torch.stack(priority).view(-1)
        td_loss_total = sum(priority)/len(priority)
        
        vis.bar(X=td_loss.view(-1,1), win= self.win_bar_td, opts=dict(title='push td_loss'))
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
            mhx = state_mem[start][0].to(dev)
            mcx = state_mem[start][1].to(dev)
            thx = state_mem[start][2].to(dev)
            tcx = state_mem[start][3].to(dev)
                
            state   =episode[0][ii:ii+seq_len+n_step]  
            action  =episode[1][ii:ii+seq_len+n_step]
            reward  =episode[2][ii:ii+seq_len+n_step]
            gamma   =episode[3][ii:ii+seq_len+n_step] 
            ireward =episode[4][ii:ii+seq_len+n_step]
            igamma  =episode[5][ii:ii+seq_len+n_step]
            
            s.append([episode_idx,ii,state,action,reward,gamma,ireward,igamma,mhx,mcx, thx,tcx ,brun_state])

        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,mhx,mcx, thx,tcx, burn_state = zip(*s)
        
        shape = (batch_size,-1)
        state   = torch.cat(state,1).to(dev)
        action   = torch.cat(action,1).to(dev)
        reward   = torch.cat(reward,1).to(dev)
        gamma   = torch.cat(gamma,1).to(dev)
        ireward   = torch.cat(ireward,1).to(dev)
        igamma   = torch.cat(igamma,1).to(dev)
        
        epi_idx = torch.LongTensor(epi_idx).reshape(shape).to(dev)
        seq_idx = torch.LongTensor(seq_idx).reshape(shape).to(dev)
        
        mhx = torch.cat(mhx,0).reshape((batch_size,1,-1 )).to(dev)
        mcx = torch.cat(mcx,0).reshape((batch_size,1,-1 )).to(dev)
        thx = torch.cat(thx,0).reshape((batch_size,1,-1 )).to(dev)
        tcx = torch.cat(tcx,0).reshape((batch_size,1,-1 )).to(dev)
        
        return epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,mhx,mcx, thx,tcx, burn_state
    
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


        vis.bar(X=torch.stack(bar), win= self.win_bar, opts=dict(title='total priority'))
        

            
            
            
#        eta = 0.9
#        p = (eta * loss.max(1) + (1.-eta)*loss.mean(1))**PER_alpha
#        priority = []
#        eta = 0.9
#        for i in range(len(td_array)-seq_len):
#            p = (eta*td_array[i:i+seq_len].max()+(1.-eta)*td_array[i:i+seq_len].mean())**PER_alpha
#            priority.append(p)
            
#        priority = torch.stack(priority).view(-1)
        
#        [data,td_loss,priority,td_loss_total,state_mem]
#        self.buffer[epi_idx][2] = priority
#        self.buffer[epi_idx][3] = td_loss_total
        
#        self.buffer[epi_idx][2] = sum(self.buffer[epi_idx][1][:])/len(self.buffer[epi_idx][1][:])
    def __len__(self):
        return self.count
    def __repr__(self):
        return '\rmem size: {}/{} ' .format(self.count, self.capacity)



class Flatten(nn.Module):
    def forward(self,inputs):
        return inputs.view(inputs.size(0),-1)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, dev ):
        super(DQN,self).__init__()
        if cnn_enable:
            size=7*7*64
            self.feature = nn.Sequential(
                    nn.Conv2d(num_inputs,64,8,stride= 4),nn.ReLU(),
                    nn.Conv2d(64,64,4,stride=2),nn.ReLU(),
                    nn.Conv2d(64,64,3,stride=1),nn.ReLU(),
                    Flatten(),
                    nn.Linear(size,256),nn.ReLU(),
                    )
        else :
            self.feature = nn.Sequential(
                    nn.Linear(s_dim,256),nn.ReLU(),
                    )

        self.lstm_size = 256
        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        
        self.advantage = nn.Sequential(
                nn.Linear(self.lstm_size,256),nn.ReLU(),
                nn.Linear(256,256),nn.ReLU(),
                nn.Linear(256,num_outputs),
                )
        self.value = nn.Sequential(
                nn.Linear(self.lstm_size,256),nn.ReLU(),
                nn.Linear(256,256),nn.ReLU(),
                nn.Linear(256,1),
                )
        self.iadvantage = nn.Sequential(
                nn.Linear(self.lstm_size,256),nn.ReLU(),
                nn.Linear(256,256),nn.ReLU(),
                nn.Linear(256,num_outputs),
                )
        self.ivalue = nn.Sequential(
                nn.Linear(self.lstm_size,256),nn.ReLU(),
                nn.Linear(256,256),nn.ReLU(),
                nn.Linear(256,1),
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
        
        adv = self.advantage(x)
        val = self.value(x)
        iadv = self.iadvantage(x)
        ival = self.ivalue(x)
        
        Q = val + adv - adv.mean()
        iQ = ival + iadv - iadv.mean()
        Qa = Q.argmax(1).view(-1,1)
        iQa = iQ.argmax(1).view(-1,1)
        return Q,Qa,iQ,iQa

    
    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return torch.zeros((1 ,self.lstm_size)).cpu(), torch.zeros((1 ,self.lstm_size)).cpu()
        else:
            return self.hx.detach().cpu(), self.cx.detach().cpu()
        

class RND(nn.Module):
    def __init__(self,num_inputs):
        super(RND,self).__init__()
        self.target= nn.Sequential(
                nn.Conv2d(num_inputs*2,64,8,stride=4),nn.ReLU(),
                nn.Conv2d(64,64,4,stride=2),nn.ReLU(),
                nn.Conv2d(64,64,3,stride=1),nn.ReLU(),
                Flatten(),
                nn.Linear(64*7*7,256),nn.ReLU(),
                nn.Linear(256,256),
                )
        self.predictor = nn.Sequential(
                nn.Conv2d(num_inputs*2,64,8,stride=4),nn.ReLU(),
                nn.Conv2d(64,64,4,stride=2),nn.ReLU(),
                nn.Conv2d(64,64,3,stride=1),nn.ReLU(),
                Flatten(),
                nn.Linear(64*7*7,256),nn.ReLU(),
                nn.Linear(256,256),nn.ReLU(),
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








def update_target(tar,cur):
    tar.load_state_dict(cur.state_dict())


def calc_td(main_model,target_model,state, action, reward,gamma,ireward,igamma,mhx,mcx, thx,tcx, story_len): 
    y_t_hat = []
    iy_t_hat = []
    with torch.no_grad():
        main_model.set_state(mhx,mcx)
        target_model.set_state(thx,tcx)
        for i in range(n_step):
            _,_,_,_ = main_model(state[i])
            _,_,_,_ = target_model(state[i])
        for i in range(story_len):
            qv,_,iqv,_ = main_model(state[i+n_step])
            _,tqa,_,tiqa = target_model(state[i+n_step])

            y_t_hat.append(reward[i] + (gamma[i+n_step]**n_step)*qv.gather(1,tqa))
            iy_t_hat.append(ireward[i] + (igamma[i+n_step]**n_step)*iqv.gather(1,tiqa))
    
    losses=[]
    main_model.reset_state()
    target_model.reset_state()
    main_model.set_state(mhx,mcx)
    target_model.set_state(thx,tcx)
    for i in range(story_len):
        q,_,iq,_ = main_model(state[i])
        td = q.gather(1,action[i]) - y_t_hat[i]
        itd = iq.gather(1,action[i]) - iy_t_hat[i]
        losses.append(td+itd)
    
    return torch.cat(losses,1).abs()
        




def actor_process(a_id,num_frames,shared_state,shared_queue,block=True, eps=0.1):
    print(f'#{a_id} start')
    win_epsil = vis.line(Y=torch.tensor([0]),opts=dict(title='epsilon'+str(a_id)))
    win_r = vis.line(Y=torch.tensor([0]),opts=dict(title='reward'+str(a_id)))
    win_exp_q = vis.line(Y=torch.tensor([0]),opts=dict(title='exp_q'+str(a_id)))

          
    mainQ = DQN(s_dim, a_dim, dev ).to(dev)
    targetQ = DQN(s_dim, a_dim, dev ).to(dev)
    rnd_model  = RND(s_dim).to(dev)
    
    mainQ.load_state_dict(shared_state["mainQ"].state_dict())
    targetQ.load_state_dict(shared_state["targetQ"].state_dict())
     
    episode_reward=0
    local_mem = []
    epsilon = 1
    state_mem = []
    done = True
    gamma = 0.997
    state = env.reset()
    q_val=[]
    for frame_idx in range(num_frames):
        if done:
            
            
            if len(local_mem)!=0:
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
                    mainQ.reset_state()
                    targetQ.reset_state()
                    mhx,mcx= mainQ.get_state()
                    thx,tcx= targetQ.get_state()
                    state,action,reward,gamma,ireward,igamma = zip(*local_mem)
        
                    b_len = len(local_mem)
                    state = torch.stack(state)
                    action = torch.LongTensor(action).reshape((b_len,1,1))
                    reward = torch.Tensor(reward).reshape((b_len,1,1))
                    gamma = torch.Tensor(gamma).reshape((b_len,1,1))
                    ireward = torch.Tensor(ireward).reshape((b_len,1,1))
                    igamma = torch.Tensor(igamma).reshape((b_len,1,1))
                    
                    td_array = calc_td(mainQ.to(dev),
                                       targetQ.to(dev),
                                       state.to(dev),
                                       action.to(dev),
                                       reward.to(dev),
                                       gamma.to(dev),
                                       ireward.to(dev),
                                       igamma.to(dev) ,
                                       mhx.to(dev),
                                       mcx.to(dev),
                                       thx.to(dev),
                                       tcx.to(dev), 
                                       b_len-n_step)
                    
                    
                    blocking = True if shared_queue.qsize()>max_shared_q_size and block else False
                    shared_queue.put([[state.cpu() ,action,reward,gamma,ireward,igamma ],td_array.cpu(),state_mem],block=blocking)
                    
                if block == False:
                    return 0
        
            state = env.reset()
            episode_reward=0
            gamma = 0.997
            local_mem = []
            state_mem = []
            mainQ.reset_state()
            targetQ.reset_state()
            q_val = []
            
            
    
        
    #    epsilon= 0.01**(EPS_CONST*frame_idx/num_frames)
        epsilon= eps
       
        with torch.no_grad():
            mhx,mcx = mainQ.get_state()
            thx,tcx = targetQ.get_state()
            state_mem.append([mhx,mcx,thx,tcx])
            qv,qa,iqv,iqa = mainQ(state)
            _,_,_,_ = targetQ(state)
            
        action = qa.item() if random.random() > epsilon else random.randrange(a_dim)
        
        q_val.append(qv.detach())
#        if vis_render:
#            vis.image(state.view(84,84),win = win_img)
            
        next_state , reward, done ,_ = env.step(action)
        local_mem.append([state, action ,reward, gamma, 0 , 0])
        
        state = next_state
        episode_reward += reward
    
        
        if shared_state["update"][a_id]:
            mainQ.load_state_dict(shared_state["mainQ"].state_dict())
            targetQ.load_state_dict(shared_state["targetQ"].state_dict())
            shared_state["update"][a_id]=False
            
            print('actor_update',mainQ.value[0].weight[0][0:5])
    
    
    print('done')
    env.close()
    

    
    
def learner_process(max_id,num_frames,shared_state,shared_queue,block=True):
    win_ir = vis.line(Y=torch.tensor([0]),opts=dict(title='ireward'))
    win_l0 = vis.line(Y=torch.tensor([0]),opts=dict(title='loss'))
    win_l1 = vis.line(Y=torch.tensor([0]),opts=dict(title='rnd_loss'))
    
    mainQ = DQN(s_dim, a_dim, dev ).to(dev)
    targetQ = DQN(s_dim, a_dim, dev ).to(dev)
    rnd_model  = RND(s_dim).to(dev)
    
    mainQ.load_state_dict(shared_state["mainQ"].state_dict())
    targetQ.load_state_dict(shared_state["targetQ"].state_dict())
                    
    optimizer = optim.Adam(mainQ.parameters(),lr)
    rnd_optimizer = optim.Adam(rnd_model.parameters(),rnd_lr)
    
    
    replay_buffer = ReplayBuffer(mem_size)
    def soft_update(target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    def update():
        epi_idx,seq_idx,state, action, reward,gamma,ireward,igamma,mhx,mcx, thx,tcx, burn_state = replay_buffer.sample(batch_size)
        
        burned_hx = []
        burned_cx = []
        burned_thx = []
        burned_tcx = []
        with torch.no_grad():
            for i in range(batch_size):
                mainQ.reset_state()
                targetQ.reset_state()
                
                mainQ.set_state(mhx[i],mcx[i])
                targetQ.set_state(thx[i],tcx[i])
                
                for j in range(len(burn_state[i])):
                    _,_,_,_ = mainQ(burn_state[i][j])
                    _,_,_,_ = targetQ(burn_state[i][j])
                
                t_mhx,t_mcx = mainQ.get_state()
                burned_hx.append(t_mhx)
                burned_cx.append(t_mcx)
                
                t_thx,t_tcx = targetQ.get_state()
                burned_thx.append(t_thx)
                burned_tcx.append(t_tcx)
            
            mhx = torch.cat(burned_hx,0).to(dev)
            mcx = torch.cat(burned_cx,0).to(dev)
            thx = torch.cat(burned_thx,0).to(dev)
            tcx = torch.cat(burned_tcx,0).to(dev)
            
        loss = calc_td(mainQ,targetQ,state, action, reward,gamma,ireward,igamma,mhx,mcx, thx,tcx,seq_len) 
        optimizer.zero_grad()
        loss.pow(2).mean().backward()
        optimizer.step()
    #            pm,tm = rnd_model(state,nstate)
    #            rnd_loss = ((pm-tm)**2).mean()
    #            rnd_optimizer.zero_grad()
    #            rnd_loss.backward()
    #            rnd_optimizer.step()
        
        for i in range(len(epi_idx)):
            replay_buffer.priority_update(epi_idx[i],seq_idx[i],loss[i].detach())
        
        return loss.pow(2).mean().item(),0
    
#    if len(replay_buffer)==0:
    if block==False:
        if shared_queue.qsize()<2 :
            print('return  shared q size > 2 ')
            return 0
        data = shared_queue.get(block=True)
        replay_buffer.push(data[0],data[1],data[2])
        
    
    while len(replay_buffer) < start_frame and block:
        print(repr(replay_buffer),end='\r')
        data = shared_queue.get(block=True)
        replay_buffer.push(data[0],data[1],data[2])
    
    
    for frame_idx in range(num_frames):
        
#        if shared_queue.qsize()!=0:
        while shared_queue.qsize() != 0:
            data = shared_queue.get()
            replay_buffer.push(data[0],data[1],data[2])

        loss, rnd_loss = update()
        print(f'#learner  l:{loss:.5f}')
        vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([loss]),win=win_l0,update ='append')
        vis.line(X=torch.tensor([frame_idx]),Y=torch.tensor([rnd_loss]),win=win_l1,update ='append')
        
#        if frame_idx % 1000 == 0:
        if random.random() < 1/10 :
            update_target(targetQ,mainQ)
#        if frame_idx % 3000 == 0:
        if random.random() < 1/20 :
            shared_state["mainQ"].load_state_dict(mainQ.state_dict())
            shared_state["targetQ"].load_state_dict(targetQ.state_dict())
            for i in range(max_id):
                shared_state["update"][i]=True
        if block == False:
            return 0
                
                


if __name__ == '__main__':
    os.system('cls')
    
    vis.close()
      
    num_processes = 3
    
    shared_queue = mp.Queue()
    shared_state = dict()
    
    shared_state["mainQ"] = DQN(s_dim, a_dim, dev ).share_memory()
    shared_state["targetQ"] =  DQN(s_dim, a_dim, dev ).share_memory()
    
    shared_state["update"] = mp.Array('i', [0 for i in range(num_processes)])
    
#    actor_process(0,num_frames,shared_state,shared_queue,False)
#    for i in range(100):
#        actor_process(0,num_frames,shared_state,shared_queue,False)
#        learner_process(1,num_frames,shared_state,shared_queue,False)
    
#
#    time.sleep(10)
    
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
    
    