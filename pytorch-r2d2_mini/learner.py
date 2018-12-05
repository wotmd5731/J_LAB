import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import random
import gym
import os
from copy import deepcopy
from time import time, sleep
import torch.multiprocessing as mp

from models import ActorNet, CriticNet
import pickle

from actor import Actor, actor_process
import queue
import visdom

vis = visdom.Visdom()


ttime = time()
def time_check(num=0):
    global ttime
    print(f'{num}  time:{time()-ttime}')
    ttime = time()

class LearnerReplayMemory:
    def __init__(self, memory_sequence_size ,config, dev ):
        self.memory_sequence_size = memory_sequence_size
        self.sequence_counter = 0
        self.batch_size = config['batch_size']
        self.memory = deque()
        self.recurrent_state = deque()
        self.priority = deque()
        self.total_priority = deque()

        self.dev = dev
        self.burn_in_length = config['burn_in_length'] # 40-80
        self.learning_length = config['learning_length']
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = config['n_step']
        
        self.obs_shape = (self.sequence_length+self.n_step,self.batch_size)+config['obs_space'][1:]
        self.reward_shape = (self.sequence_length+self.n_step,self.batch_size)+config['reward_space'][1:]
        self.gamma_shape = (self.sequence_length+self.n_step,self.batch_size)+config['gamma_space'][1:]
        self.action_shape = (self.sequence_length+self.n_step,self.batch_size)+config['reward_space'][1:]
        
        
    
    def size(self):
#        return sum([len(self.memory[i]) for i in range(len(self.memory))])
        return len(self.memory)
    
    def get(self, index):
        return self.memory[index]
    
    def clear(self):
        self.memory.clear()
        self.recurrent_state.clear()
        self.priority.clear()
        self.total_priority.clear()

    def get_weighted_sample_index(self):
        ii = []
        for i in range(self.batch_size):
            ii.append(random.randint(0,len(self.memory)-1))
        return ii
#        total_priority = torch.tensor(self.total_priority).view(-1)
#        print('priority : ',total_priority.size(0))
            
            
#        return torch.utils.data.WeightedRandomSampler(total_priority, self.batch_size, replacement=True)
#        return torch.utils.data.BatchSampler(total_priority, self.batch_size ,drop_last=False)
#        return torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(total_priority), self.batch_size, drop_last=False)
    
    def sample(self):
        sample_episode_index = self.get_weighted_sample_index()
#        sample_episode_index = [index for index in sample_episode_index][0]

        # batch * sequence * elements(obs, action, reward, done)
        sample_sequence_index = []
        rnn_state_batch = []
        traj_s=[]
        traj_a=[]
        traj_r = []
        traj_gam = []

        for episode_index in sample_episode_index:
            episode_trajectory = self.memory[episode_index]
            sequence_index = random.randint(0,len(episode_trajectory)-2)
#            priority = torch.tensor(self.priority[episode_index])
#            sequence_index = torch.utils.data.BatchSampler(priority, 1,drop_last=False)
#            sequence_index = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(priority), 1, drop_last=False)
            
#            sequence_index = [index for index in sequence_index]
#            sequence_index = sequence_index[0][0]
            sample_sequence_index.append(sequence_index)
            
            ss,aa,rr,gg = zip(*(episode_trajectory[sequence_index: sequence_index + self.sequence_length+self.n_step]))

            traj_s.append(torch.cat(ss))
            traj_a.append(torch.cat(aa))
            traj_r.append(torch.cat(rr))
            traj_gam.append(torch.cat(gg))


            #trajectory_sequence_batch.append(episode_trajectory[sequence_index: sequence_index + self.sequence_length+self.n_step])
        
            #rnn_state_batch.append(episode_rnn_state[sequence_index])

        # elements(obs, action, reward, terminal) * sequence * batch
        
        #trajectory_batch_sequence = [[[trajectory_sequence_batch[b][s][e] for b in range(self.batch_size)] for s in range(self.sequence_length+self.n_step)] for e in range(4)]
# 4,18,6,1,3,,  [obs_act_rew_gam, seq, batch, ]

        obs_batch_sequence = torch.stack(traj_s).reshape( self.obs_shape ).to(self.dev)
        action_batch_sequence = torch.stack(traj_a).reshape( self.action_shape ).to(self.dev)
        reward_batch_sequence = torch.stack(traj_r).reshape( self.reward_shape ).to(self.dev)
        gamma_batch_sequence = torch.stack(traj_gam).reshape( self.gamma_shape ).to(self.dev)

#        obs_batch_sequence = torch.Tensor(trajectory_batch_sequence[0]).to(self.dev)
#        action_batch_sequence = torch.Tensor(trajectory_batch_sequence[1]).to(self.dev)
#        reward_batch_sequence = torch.Tensor(trajectory_batch_sequence[2]).to(self.dev)
#        gamma_batch_sequence = torch.Tensor(trajectory_batch_sequence[3]).to(self.dev)


        return sample_episode_index, sample_sequence_index, obs_batch_sequence, action_batch_sequence, reward_batch_sequence, gamma_batch_sequence

    def append(self, data):
        self.memory.append(data[0])
#        self.priority.append(data[1])
#        self.total_priority.append(sum(data[1]))
        
#        self.sequence_counter += sum([len(data[0]) - (self.sequence_length+self.n_step-1) ])
        self.sequence_counter += 1
        while self.sequence_counter > self.memory_sequence_size:
#            self.sequence_counter -= len(self.memory.popleft()) - (self.sequence_length)
            self.sequence_counter -= 1
            self.priority.popleft()
            self.total_priority.popleft()

                
                
                
def calc_priority(td_loss, eta=0.9):
    return td_loss
#    stack = td_loss
#    return eta* stack.max(dim=0)[0] + (1.-eta )*stack.mean(dim=0)

#    return eta * max((td_loss)) + (1. - eta) * (sum((td_loss)) / len(td_loss))       
                
                
                

def soft_update(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




class Learner:
    def __init__(self, learner_id,config,dev,shared_state,shared_queue):

        self.action_size = config['action_space']
        self.obs_size = config['obs_space']

        self.shared_queue = shared_queue
        self.shared_state = shared_state
        
        self.dev = dev
        self.id = learner_id
        self.burn_in_length = config['burn_in_length'] # 40-80
        self.learning_length = config['learning_length']
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = config['n_step']
        self.sequence = []
        self.recurrent_state = []
        self.priority = []
        self.td_loss = deque(maxlen=self.learning_length)

        self.gamma = config['gamma']
#        self.actor_parameter_update_interval = config['actor_parameter_update_interval']


        self.actor = ActorNet(dev,config).to(self.dev)
        self.target_actor = ActorNet(dev,config).to(self.dev)
        self.critic = CriticNet(dev,config).to(self.dev)
        self.target_critic = CriticNet(dev,config).to(self.dev)
        
        self.actor.load_state_dict(self.shared_state["actor"].state_dict())
        self.target_actor.load_state_dict(self.shared_state["target_actor"].state_dict())
        self.critic.load_state_dict(self.shared_state["critic"].state_dict())
        self.target_critic.load_state_dict(self.shared_state["target_critic"].state_dict())

        
#        self.actor.load_state_dict(self.shared_state["actor"])
#        self.target_actor.load_state_dict(self.shared_state["target_actor"])
#        self.critic.load_state_dict(self.shared_state["critic"])
#        self.target_critic.load_state_dict(self.shared_state["target_critic"])
        
        self.learner_actor_rate = config['learner_actor_rate']


        self.num_actors = learner_id
        self.n_actions = 1
        self.max_frame = config['learner_max_frame']
    
        self.memory_sequence_size = config['memory_sequence_size']
        self.batch_size = config['batch_size']
        self.memory = LearnerReplayMemory(self.memory_sequence_size, config, dev)

        self.model_path = './'
#        self.memory_path = './memory_data/'
#        self.model_save_interval = 10 # 50
        self.learner_parameter_update_interval = config['learner_parameter_update_interval'] # 50
        self.target_update_inverval = config['target_update_interval'] # 100

        self.gamma = config['gamma']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_criterion = nn.MSELoss()
        
        
    def __del__(self):
        self.shared_queue.close()

                    
#        self.save_model()
    
    
    def save_model(self):
        model_dict = {'actor': self.actor.state_dict(),
                      'target_actor': self.target_actor.state_dict(),
                      'critic': self.critic.state_dict(),
                      'target_critic': self.target_critic.state_dict()}
        torch.save(model_dict, self.model_path + 'model.pt')
    
    def update_target_model(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


    def run(self):
        time_check(-1)
        while  self.memory.size() < self.batch_size :
            self.memory.append(self.shared_queue.get(block=True))
#            self.memory.append(self.shared_queue.get())
            print('\rmem size: ',self.memory.size(),end='\r')
        time_check(1)    
        count_mem=0
        frame = 0
        win_v = vis.line(Y=torch.Tensor([0]), opts=dict(title ='V_loss'))
        win_p = vis.line(Y=torch.Tensor([0]), opts=dict(title ='P_loss'))
        
        
        while frame  < self.max_frame:
            
            if self.shared_queue.qsize()!=0:
                self.memory.append(self.shared_queue.get(block=True))    
            
            
            frame+=1
            
            count_mem -= 1
            
            episode_index, sequence_index, obs_seq, action_seq, reward_seq, gamma_seq = self.memory.sample()


            # update ciritic
            q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions)
            
            target_q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions)
            for i in range(self.learning_length):
                obs_i = self.burn_in_length + i
                next_obs_i = self.burn_in_length + i + self.n_step
#                q_value[i*self.batch_size: (i+1)*self.batch_size] = self.critic(obs_seq[obs_i], action_seq[obs_i])
                
                q_value[i*self.batch_size: (i+1)*self.batch_size] = self.critic(obs_seq[obs_i], action_seq[obs_i]).gather(1,action_seq[obs_i])
                
                
                next_q_value = self.target_critic(obs_seq[next_obs_i], self.target_actor(obs_seq[next_obs_i])).max(1)[0].view(-1,1)
                target_q_val = reward_seq[obs_i] +  (gamma_seq[next_obs_i]** self.n_step) * next_q_value
#                target_q_val = invertical_vf(target_q_val)
                target_q_value[i*self.batch_size: (i+1)*self.batch_size] = target_q_val
            
#            print(target_q_value.detach())
            critic_loss = self.actor_criterion(q_value, target_q_value.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            


            # update actor
#            self.actor.reset_state()
#            self.critic.reset_state()
#            actor_loss = torch.zeros(self.learning_length * self.batch_size, self.n_actions).to(self.dev)
#            for i in range(self.learning_length):
#                obs_i = i + self.burn_in_length
#                action = self.actor(obs_seq[obs_i])
#                actor_loss[i*self.batch_size: (i+1)*self.batch_size] = -self.critic(obs_seq[obs_i], self.actor(obs_seq[obs_i]))
#            actor_loss = actor_loss.mean()

#            self.actor_optimizer.zero_grad()
#            actor_loss.backward()
#            self.actor_optimizer.step()

            # update target networks
            if frame % self.target_update_inverval == 0:
                self.update_target_model()
                
                
            print('#',frame,'critic_loss:',critic_loss.item(),'  count:',count_mem)
#            win_p = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([actor_loss.item()]), win= win_p , update ='append')
            win_v = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([critic_loss.item()]), win= win_v , update ='append')
            
            
            # calc priority
#            average_td_loss = ((q_value - target_q_value)**2).detach().to(self.dev)
#            print(max(self.memory.total_priority))
#            average_td_loss = np.mean(((q_value - target_q_value)**2).detach().cpu().numpy() , axis = 1)
#            for i in range(len(episode_index)):
#                td = average_td_loss[i: -1: self.batch_size]
#                td = average_td_loss[i]
                
#                print(self.memory.priority[episode_index[i]][sequence_index[i]] , calc_priority(td).cpu(), td)
                
#                self.memory.priority[episode_index[i]][sequence_index[i]] = calc_priority(td).cpu().view(1,-1)
#                self.memory.total_priority[episode_index[i]] = torch.cat(self.memory.priority[episode_index[i]]).sum(0).view(1,-1)
                
#                self.memory.priority[episode_index[i]][sequence_index[i]] = calc_priority(td)
#                self.memory.total_priority[episode_index[i]] = sum(self.memory.priority[episode_index[i]])

#            if frame % self.model_save_interval == 0:
#                self.save_model()

            if frame % self.learner_parameter_update_interval == 0:
#                print('learner update ')
                
#                [self.shared_state["actor"][k] = v.cpu() for k,v in self.actor.state_dict().item() ]
#                [self.shared_state["target_actor"][k] = v.cpu() for k,v in self.target_actor.state_dict().item() ]
#                [self.shared_state["critic"][k] = v.cpu() for k,v in self.critic.state_dict().item() ]
#                [self.shared_state["target_critic"][k] = v.cpu() for k,v in self.target_critic.state_dict().item() ]
                    
#                
#                for k,v in self.actor.state_dict().items():
#                    self.shared_state["actor"][k] = v.cpu()
#                for k,v in self.target_actor.state_dict().items():
#                    self.shared_state["target_actor"][k] = v.cpu()
#                for k,v in self.critic.state_dict().items():
#                    self.shared_state["critic"][k] = v.cpu()
#                for k,v in self.target_critic.state_dict().items():
#                    self.shared_state["target_critic"][k] = v.cpu()
                    
#                self.shared_state["actor"] = self.actor.state_dict()
#                self.shared_state["target_actor"] = self.target_actor.state_dict()
#                self.shared_state["critic"] = self.critic.state_dict()
#                self.shared_state["target_critic"] = self.target_critic.state_dict()
                
                self.shared_state["actor"].load_state_dict(self.actor.state_dict())
                self.shared_state["critic"].load_state_dict(self.critic.state_dict())
                self.shared_state["target_actor"].load_state_dict(self.target_actor.state_dict())
                self.shared_state["target_critic"].load_state_dict(self.target_critic.state_dict())
                for i in range(self.num_actors):
                    self.shared_state["update"][i]=True
                
#                print('learner_update',self.actor.policy_l0.weight.data[0][0])

            self.actor.reset_state()
            self.target_actor.reset_state()
            self.critic.reset_state()
            self.target_critic.reset_state()



def learner_process(lid,config,dev_cpu,shared_state,shared_queue):
    learner = Learner(lid,config,dev_cpu,shared_state,shared_queue)
    learner.run()
    
from actor import Actor, actor_process
import queue
  
if __name__ == '__main__':
    config = {
            'game_name':'CartPole-v0',

            'obs_space':(1,4),
            'reward_space':(1,1),
            'gamma_space':(1,1),
            'action_space':(1,2),
            'num_envs':1,
            'use_cnn':False,
#            'action_argmax':True,
            'get_img_from_render':False,

#            'obs_space':(1,3,84,84),
#            'reward_space':(1,1),
#            'gamma_space':(1,1),
#            'action_space':(1,2),
#            'num_envs':1,
#            'use_cnn':True,
#            'action_argmax':True,
#            'get_img_from_render':True,
#            
            'action_argmax':False,
#            'game_name':'Pendulum-v0',
#            'action_space':1,
#            'obs_space':(1,3),
            'burn_in_length':0,
            'learning_length':1,
            'n_step':1,
            'memory_sequence_size':100000,
#            'actor_parameter_update_interval':2000,
            'learner_parameter_update_interval':100,
            'actor_lr':1e-4,
            'critic_lr':1e-3,
            'gamma':0.997,
            'actor_max_frame':200,
            'learner_max_frame':10,
            'batch_size':3,
            'num_processes':1,
            
            'learner_actor_rate':20,
            'target_update_interval':50,
            'max_shared_q_size':30,
            }

    num_processes = config['num_processes']
    use_cuda = torch.cuda.is_available()
    dev_cpu = torch.device('cpu')
    dev_gpu = torch.device('cuda' if use_cuda else 'cpu')

    
#    manager = mp.Manager()
#    shared_state = manager.dict()
#    shared_queue = manager.Queue()
    
    shared_queue = mp.Queue()
    
#    shared_queue = queue.Queue()
    shared_state = dict()
    

    shared_state["actor"] = ActorNet(dev_cpu,config).share_memory()
    shared_state["critic"] = CriticNet(dev_cpu,config).share_memory()
    shared_state["target_actor"] = ActorNet(dev_cpu,config).share_memory()
    shared_state["target_critic"] = CriticNet(dev_cpu,config).share_memory()
#    shared_state["frame"] = mp.Array('i', [0 for i in range(num_processes)])
#    shared_state["sleep"] = mp.Array('i', [0 for i in range(num_processes)])
    shared_state["update"] = mp.Array('i', [0 for i in range(num_processes)])
    

    
#    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu)
#    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu)
#    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu)
#    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu)
#    shared_state["frame"] = [0 for i in range(num_processes)]
#    shared_state["sleep"] = [0 for i in range(num_processes)]
#    shared_state["update"]=False
    



#
    proc_list = []
#    proc_list.append(mp.Process(target=learner_process, args=(num_processes, config,dev_gpu,shared_state,shared_queue)))
#    eps = [0.05,0.6,0.4,0.3,0.2,0.6,0.4,0.6,0.2,0.4]
#    for i in range(num_processes):
#        proc_list.append( mp.Process(target=actor_process, args=(i,config,dev_gpu,shared_state,shared_queue,eps[i])) )


#    for proc in proc_list:
#        proc.start()
        
    try:
        for i in range(10):
            actor_process(0,config,dev_cpu,shared_state,shared_queue,0.3)
#            actor_process(1,config,dev_cpu,shared_state,shared_queue,0.3)
#            actor_process(2,config,dev_cpu,shared_state,shared_queue,0.3)
            learner_process(3,config,dev_cpu,shared_state,shared_queue)
#        for proc in proc_list:
#            proc.join()
    except:
        print('qclose')
        shared_queue.close()
#        print('shared_state close')
#        shared_state["update"].close()
        
#        for key in shared_state.keys():
#            shared_state[key].close()
        print('process close')
        for proc in proc_list:
            proc.terminate()
            
            
        shared_queue.join_thread()
#        shared_state["update"].join_thread()
#        for key in shared_state.keys():
#            shared_state[key].join_thread()
#        shared_state.close()
#        shared_queue.close()
