import numpy as np
#from dm_control import suite
import torch
import torch.multiprocessing as mp

from collections import deque
#from PIL import Image
import random
import gym
import os
from time import sleep, time
from copy import deepcopy

#from replay_memory import ReplayMemory
from models import ActorNet, CriticNet

import pickle



class env_cover():
    def __init__(self,name):
        self.env = gym.make(name)
    def reset(self):
        st = self.env.reset()
        return st, 0,False
    
    def get_obs(self,obs):
        return torch.from_numpy(obs).detach().float().view(1,config['obs_space'])

    def step(self,action):
        st,rt,dt,_ = self.env.step(action)
        return st, rt, dt
    
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
        

def calc_priority(td_loss, eta=0.9):
    return eta * max((td_loss)) + (1. - eta) * (sum((td_loss)) / len(td_loss))

#def invertical_vf(x):
#    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1)
    
def h_func(x,epsilon= 10e-2):
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
def h_inv_func(x,epsilon= 10e-2):
    return torch.sign(x) * ((((torch.sqrt(1+4*epsilon*(torch.abs(x)+1+epsilon))-1)/(2*epsilon))**2)-1)    
    






class Actor:
    def __init__(self, actor_id,config,dev,shared_state,shared_queue,eps):
#        self.env = suite.load(domain_name="walker", task_name="run")
#        self.action_size = self.env.action_spec().shape[0]
#        self.obs_size = get_obs(self.env.reset().observation).shape[1]

        self.env = env_cover(config['game_name'])
        self.action_size = config['action_space']
        self.obs_size = config['obs_space']

        self.shared_queue = shared_queue
        self.shared_state = shared_state
        self.dev = dev 

        self.actor_id = actor_id
        self.burn_in_length = config['burn_in_length'] # 40-80
        self.learning_length = config['learning_length']
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = config['n_step']
        self.sequence = []
        self.recurrent_state = []
        self.priority = []
        self.td_loss = deque(maxlen=self.learning_length)
#        self.memory_sequence_size = 1000
#        self.memory = ReplayMemory(memory_sequence_size=self.memory_sequence_size)
#        self.memory_save_interval = 3
        self.max_frame = config['actor_max_frame']
        self.gamma = config['gamma']
        self.actor_parameter_update_interval = config['actor_parameter_update_interval']
        self.model_path = './'
        self.memory_path = './'
        
        self.actor = ActorNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        self.target_actor = ActorNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        self.critic = CriticNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        self.target_critic = CriticNet(config['obs_space'], config['action_space'],dev).to(self.dev)
        
        self.actor.load_state_dict(self.shared_state["actor"].state_dict())
        self.target_actor.load_state_dict(self.shared_state["target_actor"].state_dict())
        self.critic.load_state_dict(self.shared_state["critic"].state_dict())
        self.target_critic.load_state_dict(self.shared_state["target_critic"].state_dict())

#        self.actor.load_state_dict(self.shared_state["actor"])
#        self.target_actor.load_state_dict(self.shared_state["target_actor"])
#        self.critic.load_state_dict(self.shared_state["critic"])
#        self.target_critic.load_state_dict(self.shared_state["target_critic"])
        
        
#        self.load_model()
        self.epsilon = eps 
        
    def PrePro(self,obs):
        return torch.from_numpy(obs).detach().float().reshape((1,self.obs_size)).to(self.dev)
   
    def save_memory(self):
        
        model_dict = {'sequence': self.sequence,
                      'recurrent_state': self.recurrent_state,
                      'priority': self.priority,
                      }
        
        torch.save(model_dict, self.memory_path + 'memory.pt')
    
    
#    with open('outfile', 'wb') as fp:
#    pickle.dump(itemlist, fp)
#    
#    with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)
    
    
    def load_model(self):
        if os.path.isfile(self.model_path + 'model.pt'):
            while True:
                try:
                    # TODO: Delete
#                    self.actor = ActorNet(self.obs_size, self.action_size, self.actor_id%2+1).cuda().eval()
#                    self.target_actor = deepcopy(self.actor)
#                    self.critic = CriticNet(self.obs_size, self.action_size, self.actor_id%2+1).cuda().eval()
#                    self.target_critic = deepcopy(self.critic)
                    #model_dict = torch.load(self.model_path + 'model.pt', map_location={'cuda:0':'cuda:{}'.format(self.actor_id%2+1)})
                    print('waiting  model.pt')
                    model_dict = torch.load(self.model_path + 'model.pt')
                    self.actor.load_state_dict(model_dict['actor'])
                    self.target_actor.load_state_dict(model_dict['target_actor'])
                    self.critic.load_state_dict(model_dict['critic'])
                    self.target_critic.load_state_dict(model_dict['target_critic'])
                    self.actor.to(self.dev)
                    self.target_actor.to(self.dev)
                    self.critic.to(self.dev)
                    self.target_critic.to(self.dev)
                    
                except:
                    sleep(np.random.rand() * 5 + 2)
                else:
                    break

    def calc_nstep_reward(self):
        for i in range(len(self.sequence) - self.n_step):
            self.sequence[i][2] = sum([ self.sequence[i+j][2] * (self.sequence[i+j][3] ** j) for j in range(self.n_step)] )
            

    def calc_priorities(self):
        with torch.no_grad():
            self.actor.reset_state()
            self.critic.reset_state()
            self.target_actor.reset_state()
            self.target_critic.reset_state()
            self.td_loss = deque(maxlen=self.learning_length)
            self.priority = []
    
    #       이부분은  target 넷을  nstep 만큼 진행 해놓는것.
            for i in range(self.n_step):
                next_obs = self.sequence[i][0]
                next_action = self.target_actor(self.PrePro(next_obs)).to(self.dev)
                next_q_value = self.target_critic(self.PrePro(next_obs), next_action)
    
    #       n 스텝 진행 하면서 Q 벨류 예측.   seq[시퀀스][0:staet ,1:action ,2:reward,3:term->gamma]
            for i in range(len(self.sequence) - self.n_step):
    #            obs = torch.from_numpy(self.sequence[i][0]).unsqueeze(0)
                obs = self.sequence[i][0]
    #            action = self.sequence[i][1].unsqueeze(0)
                next_obs = self.sequence[i + self.n_step][0]
                action = torch.Tensor(self.sequence[i][1]).view(1,-1).to(self.dev)
    #            next_obs = torch.from_numpy(self.sequence[i + self.n_step][0]).unsqueeze(0)
                next_action = self.target_actor(self.PrePro(next_obs)).to(self.dev)
    
                q_value = self.critic(self.PrePro(obs), action)
                reward = self.sequence[i][2]
                gamma = self.sequence[i + self.n_step - 1][3]
                next_q_value = self.target_critic(self.PrePro(next_obs), next_action)
                
                if i >= self.burn_in_length:
                    target_q_value = torch.tensor(reward + (gamma ** self.n_step)).to(self.dev) * next_q_value
#                    target_q_value = invertical_vf(target_q_value)
                    self.td_loss.append(((q_value - target_q_value)**2).item())
                if i >= self.sequence_length:
                    self.priority.append(calc_priority(self.td_loss))
            
            
            

    
    def run(self):
        frame = 0
        
        while frame  < self.max_frame:
#            self.shared_state['frame'][self.actor_id]=frame
#            while self.shared_state['sleep'][self.actor_id] :
#                sleep(0.5)
            
            st, rt, dt  = self.env.reset()
            
            self.actor.reset_state()
            self.critic.reset_state()
            self.target_actor.reset_state()
            self.target_critic.reset_state()
            
            self.sequence = []
            self.recurrent_state = []
            self.priority = []
            
            self.td_loss.clear()
            
            reward_sum = 0
            count_step = 0     

            while not dt:
                frame+=1
                # get recurrent state
                actor_hx, actor_cx = self.actor.get_state()
                target_actor_hx, target_actor_cx = self.target_actor.get_state()
                critic_hx, critic_cx = self.critic.get_state()
                target_critic_hx, target_critic_cx = self.target_critic.get_state()
                
                action = self.actor(self.PrePro(st))
                target_action = self.target_actor(self.PrePro(st))
                _ = self.critic(self.PrePro(st), action)
                _ = self.target_critic(self.PrePro(st), target_action)

                action = action.detach().item() +  np.random.normal(0, self.epsilon, (self.action_size))
                action = np.clip(action, -1, 1)

                st_1, rt, dt = self.env.step(action)
    
                reward_sum += rt
                count_step += 1
                
                gamma = self.gamma if not dt else 0.
                self.sequence.append([st, action, rt, gamma])
                st = st_1

                self.recurrent_state.append([[actor_hx[0], actor_cx[0]], [target_actor_hx[0], target_actor_cx[0]], 
                                                [critic_hx[0], critic_cx[0]], [target_critic_hx[0], target_critic_cx[0]]])

                if frame % self.actor_parameter_update_interval == 0:
                    print('actor_update',self.actor.l1.weight.data[0])
#                    self.actor.load_state_dict(self.shared_state["actor"])
#                    self.target_actor.load_state_dict(self.shared_state["target_actor"])
#                    self.critic.load_state_dict(self.shared_state["critic"])
#                    self.target_critic.load_state_dict(self.shared_state["target_critic"])
        
                    self.actor.load_state_dict(self.shared_state["actor"].state_dict())
                    self.target_actor.load_state_dict(self.shared_state["target_actor"].state_dict())
                    self.critic.load_state_dict(self.shared_state["critic"].state_dict())
                    self.target_critic.load_state_dict(self.shared_state["target_critic"].state_dict())
#                    self.load_model()


            if len(self.sequence) >= self.sequence_length:
                self.sequence.extend([(st, action, 0., 0.) for i in range(self.n_step)])
                self.calc_nstep_reward()
                self.calc_priorities()
                
#                while self.shared_state['data'][self.actor_id]:
#                    sleep(0.1)
                
#                self.shared_state['data'][self.actor_id]=True
#                with open('actor{}.mt'.format(self.actor_id), 'wb') as f:
#                    pickle.dump([self.sequence, self.recurrent_state, self.priority], f)
                
                
                
#                while self.shared_queue.qsize() > 100:
#                    print('shared Queue  sleep')
#                    time.sleep(1)
                self.shared_queue.put([self.sequence, self.recurrent_state, self.priority],block=True)
                
#                print(len(self.sequence),len(self.recurrent_state),len(self.priority))
#                self.memory.add(self.sequence, self.recurrent_state, self.priority)
                
#            if self.actor_id == 0:
            print('#',self.actor_id,'frame:', frame,'step:', count_step, 'reward:', reward_sum)
                      
            
#            if len(self.memory.memory) > self.memory_save_interval:
#                self.memory.save(self.actor_id)

def actor_process(actor_id,config,dev_cpu,shared_state,shared_queue,eps):
    actor = Actor(actor_id,config,dev_cpu,shared_state,shared_queue,eps)
    actor.run()
    
    
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
            'actor_max_frame':200,
            'learner_max_frame':1,
            'batch_size':32,
            }


    use_cuda = torch.cuda.is_available()
    dev_cpu = torch.device('cpu')
    dev_gpu = torch.device('cuda' if use_cuda else 'cpu')
    
#    manager = mp.Manager()
#    shared_state = manager.dict()
    shared_state = dict()
#    shared_queue = manager.Queue()
    shared_queue = mp.Queue()
    num_processes = 5
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
    
    
#    for i in range(10):
#        actor_process(0,config,dev_cpu,shared_state,shared_queue)
#        actor_process(1,config,dev_cpu,shared_state,shared_queue)
#        actor_process(2,config,dev_cpu,shared_state,shared_queue)
#        learner_process(1,config,dev_cpu,shared_state,shared_queue)



#    learner_procs = mp.Process(target=learner_process, args=(0, config,dev_gpu,shared_state,shared_queue))
#    learner_procs.start()
    
    actor_procs = []
    for i in range(1, num_processes):
        actor_proc = mp.Process(target=actor_process, args=(i,config,dev_cpu,shared_state,shared_queue))
        actor_proc.start()
        actor_procs.append(actor_proc)

#    learner_procs.join()
    actor_procs[0].join()