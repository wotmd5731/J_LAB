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
#mp.set_start_method('spawn', force=True)
from models import ActorNet, CriticNet
import queue
import visdom

vis = visdom.Visdom()

os.system('cls')


from actor import Actor, actor_process
from learner import Learner, learner_process
from models import ActorNet, CriticNet

if __name__ == '__main__':
    vis.close()
    
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
            'memory_sequence_size':1000000,
#            'actor_parameter_update_interval':2000,
            'learner_parameter_update_interval':50,
            'actor_lr':1e-4,
            'critic_lr':1e-4,
            'gamma':0.997,
            'actor_max_frame':1000000,
            'learner_max_frame':100000,
            'batch_size':64,
            'num_processes':1,
            
            'learner_actor_rate':20,
            'target_update_interval':50,
            'max_shared_q_size':5,
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
    
#    for i in range(10):
#        actor_process(0,config,dev_cpu,shared_state,shared_queue,0.3)
#        actor_process(1,config,dev_cpu,shared_state,shared_queue,0.3)
#        actor_process(2,config,dev_cpu,shared_state,shared_queue,0.3)
#        learner_process(1,config,dev_cpu,shared_state,shared_queue)


#
    proc_list = []
    proc_list.append(mp.Process(target=learner_process, args=(num_processes, config,dev_cpu,shared_state,shared_queue)))
    eps = [0.10,0.6,0.4,0.3,0.2,0.6,0.4,0.6,0.2,0.4]
    for i in range(num_processes):
        proc_list.append( mp.Process(target=actor_process, args=(i,config,dev_cpu,shared_state,shared_queue,eps[i])) )


    for proc in proc_list:
        proc.start()
        
    try:
        for proc in proc_list:
            proc.join()
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
        
            
