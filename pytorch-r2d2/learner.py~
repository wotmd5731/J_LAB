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
        total_priority = torch.tensor(self.total_priority).view(-1)
#        print('priority : ',total_priority.size(0))
            
            
        return torch.utils.data.WeightedRandomSampler(total_priority, self.batch_size, replacement=False)
    
    def sample(self):
        # エピソードのインデックスを取得
        sample_episode_index = self.get_weighted_sample_index()
        sample_episode_index = [index for index in sample_episode_index]

        # 各エピソードの中からサンプルするシーケンスのインデックスを取得
        # batch * sequence * elements(obs, action, reward, done)
        sample_sequence_index = []
        trajectory_sequence_batch = []
        rnn_state_batch = []
        for episode_index in sample_episode_index:
            episode_trajectory = self.memory[episode_index]
            priority = torch.tensor(self.priority[episode_index])
            sequence_index = torch.utils.data.WeightedRandomSampler(priority, 1, replacement = False)
            sequence_index = [index for index in sequence_index]
            sequence_index = sequence_index[0]
            sample_sequence_index.append(sequence_index)
            trajectory_sequence_batch.append(episode_trajectory[sequence_index: sequence_index + self.sequence_length+self.n_step])

            episode_rnn_state = self.recurrent_state[episode_index]
            rnn_state_batch.append(episode_rnn_state[sequence_index])

        # elements(obs, action, reward, terminal) * sequence * batch
        
        trajectory_batch_sequence = [[[trajectory_sequence_batch[b][s][e] for b in range(self.batch_size)] for s in range(self.sequence_length+self.n_step)] for e in range(4)]
#        4,18,6,1,3,,  [obs_act_rew_gam, seq, batch, ]
        obs_batch_sequence = torch.Tensor(trajectory_batch_sequence[0]).to(self.dev)
        action_batch_sequence = torch.Tensor(trajectory_batch_sequence[1]).to(self.dev)
        reward_batch_sequence = torch.Tensor(trajectory_batch_sequence[2]).to(self.dev)
        gamma_batch_sequence = torch.Tensor(trajectory_batch_sequence[3]).to(self.dev)

        # batch * state -> state * batch  ,4,2,6,128
        rnn_state_batch = torch.stack([torch.stack([ torch.stack([rnn_state_batch[b][e][i] for b in range(self.batch_size)]) for i in range(2)]) for e in range(4)])
        actor_state_batch = torch.Tensor(rnn_state_batch[0]).to(self.dev)
        target_actor_state_batch = torch.Tensor(rnn_state_batch[1]).to(self.dev)
        critic_state_batch = torch.Tensor(rnn_state_batch[2]).to(self.dev)
        target_critic_state_batch = torch.Tensor(rnn_state_batch[3]).to(self.dev)

        return sample_episode_index, sample_sequence_index, obs_batch_sequence, action_batch_sequence, reward_batch_sequence, gamma_batch_sequence, \
                actor_state_batch, target_actor_state_batch, critic_state_batch, target_critic_state_batch

    def append(self, data):
        self.memory.append(data[0])
        self.recurrent_state.append(data[1])
        self.priority.append(data[2])
        self.total_priority.append(sum(data[2]))

#        self.sequence_counter += sum([len(data[0]) - (self.sequence_length+self.n_step-1) ])
        self.sequence_counter += 1
        while self.sequence_counter > self.memory_sequence_size:
#            self.sequence_counter -= len(self.memory.popleft()) - (self.sequence_length)
            self.sequence_counter -= 1
            self.recurrent_state.popleft()
            self.priority.popleft()
            self.total_priority.popleft()

                
                
                
def calc_priority(td_loss, eta=0.9):
    return eta * max((td_loss)) + (1. - eta) * (sum((td_loss)) / len(td_loss))       
                
                
                

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
        self.actor_parameter_update_interval = config['actor_parameter_update_interval']
        
        self.actor = ActorNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        self.target_actor = deepcopy(self.actor).to(self.dev)
        self.critic = CriticNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        self.target_critic = deepcopy(self.critic).to(self.dev)
        
        self.actor.load_state_dict(self.shared_state["actor"].state_dict())
        self.target_actor.load_state_dict(self.shared_state["target_actor"].state_dict())
        self.critic.load_state_dict(self.shared_state["critic"].state_dict())
        self.target_critic.load_state_dict(self.shared_state["target_critic"].state_dict())
#        
#        self.actor.load_state_dict(self.shared_state["actor"])
#        self.target_actor.load_state_dict(self.shared_state["target_actor"])
#        self.critic.load_state_dict(self.shared_state["critic"])
#        self.target_critic.load_state_dict(self.shared_state["target_critic"])
        
        
        
        self.n_actions = 1
        self.max_frame = config['learner_max_frame']
    
        self.memory_sequence_size = config['memory_sequence_size']
        self.batch_size = config['batch_size']
        self.memory = LearnerReplayMemory(self.memory_sequence_size, config, dev)

        self.model_path = './'
#        self.memory_path = './memory_data/'
#        self.model_save_interval = 10 # 50
        self.learner_parameter_update_interval = config['learner_parameter_update_interval'] # 50
        self.target_update_inverval = 500 # 100

        self.gamma = config['gamma']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_criterion = nn.MSELoss()
        
        
                    
                    
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
#        while len(global_buf) < train_start_size:
##        print(f'\r g_buf len :{len(global_buf)}/{train_start_size}',end='\r')
#            global_buf.append(shared_queue.get())
#            schedule_dict['step'][rank]=len(global_buf)
#    
#        print(self.shared_queue.qsize())
#        
        while  self.memory.size() < self.batch_size :
#            self.memory.append(self.shared_queue.get(block=True))
            self.memory.append(self.shared_queue.get())
            
            
#            print('waiting  shared q {}/{}'.format(self.memory.size(),self.batch_size))
#            for i in range(self.shared_queue.qsize()):
             
        
#        while True:
#            sleep(0.4)
#            count = [self.shared_state['data'][i] for i in range(self.n_actor)]
#            if sum(count) == self.n_actor:
#                break
            
            
        count_mem=0
        frame = 0
        while frame  < self.max_frame:
            
#            if frame %10 ==0:
#                for i in range(self.n_actor):
#                    if self.shared_state['data'][i]==True:
#                        with open('actor{}.mt'.format(self.actor_id), 'rb') as f:
#                            data = pickle.load(f)
#                            self.memory.append(data)
#                        self.shared_state['data'][i]=False
                    
                
#            print('waiting  shared q {}/{}'.format(self.memory.size(),self.batch_size))
            
#            self.shared_state['frame'][self.id]=frame
#            while self.shared_state['sleep'][self.id] :
#                sleep(0.5)
            if self.shared_queue.qsize()==0 and count_mem <0:
                self.memory.append(self.shared_queue.get(block=True))
#                self.memory.append(self.shared_queue.get())
                
            for i in range(self.shared_queue.qsize()):
##                global_buf.append(self.shared_queue.get())
                self.memory.append(self.shared_queue.get())
                count_mem += 5
            frame+=1
            
            count_mem -= 1
            
            episode_index, sequence_index, obs_seq, action_seq, reward_seq, gamma_seq, a_state, ta_state, c_state, tc_state = self.memory.sample()

            self.actor.set_state(a_state[0], a_state[1])
            self.target_actor.set_state(ta_state[0], ta_state[1])
            self.critic.set_state(c_state[0], c_state[1])
            self.target_critic.set_state(tc_state[0], tc_state[1])

            ### burn-in step ###
            _ = [self.actor(obs_seq[i]) for i in range(self.burn_in_length)]
            _ = [self.critic(obs_seq[i],action_seq[i]) for i in range(self.burn_in_length)]
            _ = [self.target_actor(obs_seq[i]) for i in range(self.burn_in_length+self.n_step)]
            _ = [self.target_critic(obs_seq[i],action_seq[i]) for i in range(self.burn_in_length+self.n_step)]
            ### learning steps ###

            # update ciritic
            q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions)
            target_q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions)
            for i in range(self.learning_length):
                obs_i = self.burn_in_length + i
                next_obs_i = self.burn_in_length + i + self.n_step
                q_value[i*self.batch_size: (i+1)*self.batch_size] = self.critic(obs_seq[obs_i], action_seq[obs_i])
                with torch.no_grad():
                    next_q_value = self.target_critic(obs_seq[next_obs_i], self.target_actor(obs_seq[next_obs_i]))
                    target_q_val = reward_seq[obs_i] +  (gamma_seq[next_obs_i-1]** self.n_step) * next_q_value.view(-1)
    #                target_q_val = invertical_vf(target_q_val)
                    target_q_value[i*self.batch_size: (i+1)*self.batch_size] = target_q_val.view(-1,1)
            
            critic_loss = self.actor_criterion(q_value, target_q_value.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            


            # update actor
            self.actor.reset_state()
            self.critic.reset_state()
            actor_loss = torch.zeros(self.learning_length * self.batch_size, self.n_actions).to(self.dev)
            for i in range(self.learning_length):
                obs_i = i + self.burn_in_length
                action = self.actor(obs_seq[obs_i])
                actor_loss[i*self.batch_size: (i+1)*self.batch_size] = -self.critic(obs_seq[obs_i], self.actor(obs_seq[obs_i]))
            actor_loss = actor_loss.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            if frame % self.target_update_inverval == 0:
                self.update_target_model()
                
                
            print('#',frame,'critic_loss:',critic_loss.item(),'  actor_loss:',actor_loss.item())
            
            
            # calc priority
            average_td_loss = np.mean(((q_value - target_q_value)**2).detach().cpu().numpy() , axis = 1)
            for i in range(len(episode_index)):
                td = average_td_loss[i: -1: self.batch_size]
                self.memory.priority[episode_index[i]][sequence_index[i]] = calc_priority(td)
                self.memory.total_priority[episode_index[i]] = sum(self.memory.priority[episode_index[i]])

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
                print('learner_update',self.actor.l1.weight.data[0])
                
                self.shared_state["actor"].load_state_dict(self.actor.state_dict())
                self.shared_state["critic"].load_state_dict(self.critic.state_dict())
                self.shared_state["target_actor"].load_state_dict(self.target_actor.state_dict())
                self.shared_state["target_critic"].load_state_dict(self.target_critic.state_dict())
#                self.save_model()
    
#                for i in range(self.n_actors):
#                    is_memory = os.path.isfile(self.memory_path + '/memory{}.pt'.format(i))
#                    if is_memory:
#                        self.memory.load(i)
#                    sleep(0.1)

            self.actor.reset_state()
            self.target_actor.reset_state()
            self.critic.reset_state()
            self.target_critic.reset_state()



def learner_process(lid,config,dev_cpu,shared_state,shared_queue):
    learner = Learner(lid,config,dev_cpu,shared_state,shared_queue)
    learner.run()
    
from actor import Actor, actor_process

    

if __name__ == '__main__':
    config = {
            'game_name':'Pendulum-v0',
            'action_space':1,
            'obs_space':(3),
            'burn_in_length':20,
            'learning_length':40,
            'n_step':5,
            'memory_sequence_size':1000000,
            'actor_parameter_update_interval':600,
            'gamma':0.997,
            'actor_max_frame':1000,
            'learner_max_frame':10,
            'batch_size':10,
            }


    use_cuda = torch.cuda.is_available()
    dev_cpu = torch.device('cpu')
    dev_gpu = torch.device('cuda' if use_cuda else 'cpu')
    
#    manager = mp.Manager()
#    shared_state = manager.dict()
    shared_state = dict()
#    shared_queue = manager.Queue()
    shared_queue = mp.Queue()
    num_processes = 2
    #shared_state["Q_state"]
    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
    
#    
#    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
    
    
    shared_state["frame"] = mp.Array('i', [0 for i in range(num_processes)])
    shared_state["sleep"] = mp.Array('i', [0 for i in range(num_processes)])
#    shared_state["frame"] = [0 for i in range(num_processes)]
#    shared_state["sleep"] = [0 for i in range(num_processes)]
    
    learner = Learner(9,config,dev_cpu,shared_state,shared_queue)
    
    
    for i in range(10):
        actor_process(0,config,dev_cpu,shared_state,shared_queue)
        actor_process(1,config,dev_cpu,shared_state,shared_queue)
        actor_process(2,config,dev_cpu,shared_state,shared_queue)
        learner.run()


#    learner_procs = mp.Process(target=learner_process, args=(0, config,dev_gpu,shared_state,shared_queue))
#    learner_procs.start()
#    
#    actor_procs = []
#    for i in range(1, num_processes):
#        actor_proc = mp.Process(target=actor_process, args=(i,config,dev_cpu,shared_state,shared_queue))
#        actor_proc.start()
#        actor_procs.append(actor_proc)
#
##    learner_procs.join()
#    actor_procs[0].join()