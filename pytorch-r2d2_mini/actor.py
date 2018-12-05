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
import queue
from multiprocessing_env import SubprocVecEnv
import visdom

vis = visdom.Visdom()



class env_cover():
    def __init__(self,config,dev):

        
        self.dev = dev
        self.num_env =config['num_envs']
        self.get_img_from_render = config['get_img_from_render']
        
        
        
        self.obs_shape = (self.num_env,)+config['obs_space'][1:]
#        print(self.obs_shape)
        self.reward_shape = (self.num_env,)+config['reward_space'][1:]
        self.gamma_shape = (self.num_env,)+config['gamma_space'][1:]
        
        
        
        if self.num_env == 1:
            self.env = gym.make(config['game_name'])
        else:
            def make_env():
                def _thunk():
                    env = gym.make(config['game_name'])
                    return env
                return _thunk
            envs = [make_env() for i in range(self.num_env)]
            self.env = SubprocVecEnv(envs)

#
#def obs_preproc(x):
#    if IMG_GET_RENDER ==False:  
#        return torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)
#    x = np.dot(x, np.array([[0.299, 0.587, 0.114]]).T)
#    x = np.reshape(x, (1,x.shape[1], x.shape[0]))
#    return torch.from_numpy(np.resize(x, feature_state)).float().unsqueeze(0)/255
#


    def reset(self):
        st = self.env.reset()
        if self.get_img_from_render:
            st = self.env.render(mode='rgb_array')
            st = np.resize(st,self.obs_shape)/255.
            
        
        return torch.FloatTensor(st).reshape(self.obs_shape).to(self.dev), torch.zeros(self.reward_shape).to(self.dev),torch.zeros(self.gamma_shape).to(self.dev)
        #return st, 0,False
    
#    def get_obs(self,obs):
#        return torch.from_numpy(obs).detach().float().view(1,config['obs_space'])

    def step(self,action):
        
        st,rt,dt,_ = self.env.step(action)
        
        if self.get_img_from_render:
            st = self.env.render(mode='rgb_array')
            st = np.resize(st,self.obs_shape)/255.
            
#        print(st)
        st = torch.FloatTensor(st).reshape(self.obs_shape).to(self.dev)
        rt = torch.FloatTensor([rt]).reshape(self.reward_shape).to(self.dev)
        if self.num_env ==1:
            dt = torch.FloatTensor([dt]).reshape(self.gamma_shape).to(self.dev)
        else :
            dt = torch.FloatTensor(dt.astype(int)).reshape(self.gamma_shape).to(self.dev)


        return st, rt, dt
    def end_dummy(self):
        return torch.zeros(self.obs_shape).to(self.dev), torch.zeros(self.reward_shape).to(self.dev),torch.zeros(self.gamma_shape).to(self.dev)
    
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
        

def calc_priority(td_loss, eta=0.9):
    return td_loss[0]
#    stack = torch.stack(td_loss)
#    return eta* stack.max(dim=0)[0] + (1.-eta )*stack.mean(dim=0)

#    return eta * max((td_loss)) + (1. - eta) * (sum((td_loss)) / len(td_loss))

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

        self.env = env_cover(config,dev)
        self.num_env = config['num_envs']
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
#        self.actor_parameter_update_interval = config['actor_parameter_update_interval']
        self.max_shared_q_size=config['max_shared_q_size']
        
        self.model_path = './'
        self.memory_path = './'
        
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
        self.action_argmax = config['action_argmax']
        
#        self.load_model()
        self.epsilon = eps 
    def __del__(self):
        self.env.close()
        
    def PrePro(self,obs):
        return obs
#        return torch.from_numpy(obs).detach().float().reshape((1,self.obs_size)).to(self.dev)
   
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
#            self.td_loss = deque(maxlen=self.learning_length)
            self.td_loss = []
            self.priority = []
    
    #       이부분은  target 넷을  nstep 만큼 진행 해놓는것.
#            for i in range(self.n_step):
#                next_obs = self.sequence[i][0]
#                next_action = self.target_actor(self.PrePro(next_obs)).to(self.dev)
#                next_q_value = self.target_critic(self.PrePro(next_obs), next_action)
    
    #       n 스텝 진행 하면서 Q 벨류 예측.   seq[시퀀스][0:staet ,1:action ,2:reward,3:term->gamma]
            for i in range(len(self.sequence) - self.n_step):
    #            obs = torch.from_numpy(self.sequence[i][0]).unsqueeze(0)
#                obs = self.sequence[i][0]
#    #            action = self.sequence[i][1].unsqueeze(0)
#                next_obs = self.sequence[i + self.n_step][0]
#
#                action = self.sequence[i][1]
##                action = torch.Tensor(self.sequence[i][1]).view(1,-1).to(self.dev)
#    #            next_obs = torch.from_numpy(self.sequence[i + self.n_step][0]).unsqueeze(0)
#                next_action = self.target_actor(self.PrePro(next_obs)).to(self.dev)
#    
#                q_value = self.critic(self.PrePro(obs), action)
#                q_value = q_value.gather(1,action.view(1,-1))
#                reward = self.sequence[i][2]
#                gamma = self.sequence[i + self.n_step - 1][3]
#                next_q_value = self.target_critic(self.PrePro(next_obs),next_action).max(1)[0]
#                
#                if i >= self.burn_in_length:
#                    target_q_value = (reward + (gamma ** self.n_step)) * next_q_value
##                    target_q_value = invertical_vf(target_q_value)
#                    self.td_loss.append(((q_value - target_q_value)**2))
#                    if len(self.td_loss) > self.learning_length:
#                        self.td_loss.pop(0)

#                if i >= self.sequence_length:
#                    self.priority.append(calc_priority(self.td_loss))
                self.priority.append(torch.Tensor([0]))
            
            
            
    
    def run(self):
#        sleep(random.random()*1)
        frame = 0
#        if self.actor_id%3 == 0:
        win_r = vis.line(Y=torch.Tensor([0]), opts=dict(title ='reward'+str(self.epsilon)))
        reward_sum = 0
        
        
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
#            if self.actor_id%3 == 0:
            win_r = vis.line(X=torch.Tensor([frame]), Y=torch.Tensor([reward_sum]), win= win_r , update ='append')
            qmin = 9999
            qmax = -9999
            pmin = 9999
            pmax = -9999
            
            reward_sum = 0
            count_step = 0     
            sleep(0.01)
            while sum(dt)!=self.num_env:
                
                frame+=1
                # get recurrent state
                
                action = self.actor(self.PrePro(st))
                Qv = self.critic(self.PrePro(st), action)
                qmax = max(qmax,Qv.max())
                qmin = min(qmin,Qv.min())
                pmax = max(pmax,action.max())
                pmin = min(pmin,action.min())
                
                
                
#                noise = torch.normal(mean=torch.zeros([self.num_env,1]),std=torch.ones([self.num_env,1])).to(self.dev)
#                action = action.detach().item() +  np.random.normal(0, self.epsilon, (self.action_size))
#                action = np.clip(action, -1, 1)
                action = Qv.argmax().view(1,-1)
                if self.epsilon>random.random():
                    action = torch.LongTensor([random.randint(0,1)]).view(1,-1)
                    
#                m = torch.distributions.MultivariateNormal(torch.zeros([1,1]), torch.eye(1))
#                action = action + m.sample().to(self.dev)*self.epsilon
##                action  = action.clamp(min=0,max=1)
#                
#                if self.action_argmax:
#                    act = action.argmax(1).cpu().numpy().item()
#                else:
#                    act = action.cpu().numpy()
                
#                action = (action+noise*self.epsilon).clamp(min=-1,max=1)

                st_1, rt, dt = self.env.step(int(action.item()))
                
                reward_sum += rt
                count_step += 1
                gamma = torch.ones([self.num_env,1]).to(self.dev)*self.gamma*(1-dt)
#                gamma = self.gamma if not dt else 0.
                self.sequence.append([st, action, rt, gamma])
                st = st_1

#                self.recurrent_state.append([torch.cat([actor_hx, actor_cx]), torch.cat([target_actor_hx, target_actor_cx]), 
#                                                torch.cat([critic_hx, critic_cx]), torch.cat([target_critic_hx, target_critic_cx])])

#                if True:
                if self.shared_state["update"][self.actor_id]:
                    
                    
                    self.actor.load_state_dict(self.shared_state["actor"].state_dict())
                    self.target_actor.load_state_dict(self.shared_state["target_actor"].state_dict())
                    self.critic.load_state_dict(self.shared_state["critic"].state_dict())
                    self.target_critic.load_state_dict(self.shared_state["target_critic"].state_dict())
                    self.shared_state["update"][self.actor_id]=False
#                    print('actor_update',self.actor.policy_l0.weight.data[0][0])
#                    self.load_model()


            if len(self.sequence) >= self.sequence_length:
#                self.sequence.extend([(st, action, 0., 0.) for i in range(self.n_step)])
#                st, rt, dt = self.env.end_dummy()
#                self.sequence.extend([[st,action, rt, dt] for i in range(self.n_step)])
                st, rt, dt = self.env.end_dummy()
                self.sequence.extend([[st,action, rt, dt] for i in range(self.n_step)])
                
#                self.calc_nstep_reward()
#                self.calc_priorities()
                
                
                for i in range(len(self.sequence)):
                    for j in range(4):
                        self.sequence[i][j] = self.sequence[i][j].cpu()
#                for i in range(len(self.recurrent_state)):
#                    for j in range(4):
#                        self.recurrent_state[i][j] = self.recurrent_state[i][j].cpu()
                for i in range(len(self.priority)):
                    self.priority[i] = self.priority[i].cpu()
                blocking = True if self.shared_queue.qsize()>self.max_shared_q_size else False
                self.shared_queue.put([self.sequence],block=blocking)
                
#            if self.actor_id == 0:
            print('\r#',self.actor_id,'frame:', frame,'step:', count_step, 'reward: {:.3f}'.format(reward_sum.item()), 'qmin,max :{:.3f},{:.3f},  pminmax : {:.3f},{:.3f}'.format(qmin,qmax,pmin,pmax),end='\r')
                      
            
#            if len(self.memory.memory) > self.memory_save_interval:
#                self.memory.save(self.actor_id)

def actor_process(actor_id,config,dev_cpu,shared_state,shared_queue,eps):
    with torch.no_grad():
        actor = Actor(actor_id,config,dev_cpu,shared_state,shared_queue,eps)
        actor.run()
        

    
if __name__ == '__main__':
#    config = {
#            'game_name':'CartPole-v0',
#
#            'obs_space':(1,4),
#            'reward_space':(1,1),
#            'gamma_space':(1,1),
#            'action_space':(1,2),
#            'num_envs':1,
#            'use_cnn':False,
#            'action_argmax':True,
#            'get_img_from_render':False,
#            
##            'obs_space':(1,3,84,84),
##            'reward_space':(1,1),
##            'gamma_space':(1,1),
##            'num_envs':1,
##            'use_cnn':True,
##            'action_argmax':True,
##            'get_img_from_render':True,
##            
##            'game_name':'Pendulum-v0',
##            'action_space':1,
##            'obs_space':(1,3),
#            'burn_in_length':0,
#            'learning_length':1,
#            'n_step':1,
#            'memory_sequence_size':500,
##            'actor_parameter_update_interval':2000,
#            'learner_parameter_update_interval':100,
#            'actor_lr':1e-4,
#            'critic_lr':1e-3,
#            'gamma':0.997,
#            'actor_max_frame':400,
#            'learner_max_frame':200000,
#            'batch_size':64,
#            'num_processes':1,
#            
#            'learner_actor_rate':20,
#            'target_update_interval':30,
#            'max_shared_q_size':10,
#            }
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
            'memory_sequence_size':10000,
#            'actor_parameter_update_interval':2000,
            'learner_parameter_update_interval':30,
            'actor_lr':1e-4,
            'critic_lr':1e-3,
            'gamma':0.997,
            'actor_max_frame':1000000,
            'learner_max_frame':100000,
            'batch_size':64,
            'num_processes':2,
            
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
            actor_process(1,config,dev_cpu,shared_state,shared_queue,0.3)
            actor_process(2,config,dev_cpu,shared_state,shared_queue,0.3)
#        learner_process(1,config,dev_cpu,shared_state,shared_queue)
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
