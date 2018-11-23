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

from actor import Actor, actor_process
from learner import Learner, learner_process
from models import ActorNet, CriticNet

if __name__ == '__main__':
    config = {
            'game_name':'Pendulum-v0',
            'action_space':1,
            'obs_space':(3),
            'burn_in_length':10,
            'learning_length':20,
            'n_step':5,
            'memory_sequence_size':100,
            'actor_parameter_update_interval':2000,
            'learner_parameter_update_interval':30,
            'actor_lr':1e-3,
            'critic_lr':1e-2,
            'gamma':0.997,
            'actor_max_frame':4000000,
            'learner_max_frame':200000,
            'batch_size':16,
            'num_processes':8,
            }

    num_processes = config['num_processes']
    use_cuda = torch.cuda.is_available()
    dev_cpu = torch.device('cpu')
    dev_gpu = torch.device('cuda' if use_cuda else 'cpu')
    
    
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_queue = manager.Queue()
#    shared_queue = queue.Queue()
#    shared_state = dict()
    
    #shared_state["Q_state"]
#    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
#    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
#    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()
#    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).share_memory()

    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu)
    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu)
    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu)
    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu)
    
#    shared_state["actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["target_actor"] = ActorNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
#    shared_state["target_critic"] = CriticNet(config['obs_space'], config['action_space'],dev_cpu).state_dict()
    
    
#    shared_state["frame"] = mp.Array('i', [0 for i in range(num_processes)])
#    shared_state["sleep"] = mp.Array('i', [0 for i in range(num_processes)])
    shared_state["frame"] = [0 for i in range(num_processes)]
    shared_state["sleep"] = [0 for i in range(num_processes)]
    
    
#    for i in range(10):
#        actor_process(0,config,dev_cpu,shared_state,shared_queue,0.3)
#        actor_process(1,config,dev_cpu,shared_state,shared_queue,0.3)
#        actor_process(2,config,dev_cpu,shared_state,shared_queue,0.3)
#        learner_process(1,config,dev_cpu,shared_state,shared_queue)


#
    learner_procs = mp.Process(target=learner_process, args=(0, config,dev_gpu,shared_state,shared_queue))
    learner_procs.start()
    
    eps = [0.05,0.6,0.4,0.3,0.2,0.6,0.4,0.6,0.2,0.4]
    actor_procs = []
    for i in range(num_processes):
        actor_proc = mp.Process(target=actor_process, args=(i,config,dev_cpu,shared_state,shared_queue,eps[i]))
        actor_proc.start()
        actor_procs.append(actor_proc)


#    learner_process(0, config,dev_gpu,shared_state,shared_queue)
    
    
    
    learner_procs.join()
    for act in actor_procs:
        act.join()
    
#    frame = [0 for i in range(num_processes)]
#    while True:
#        sleep(0.5)
#        for i in range(num_processes):
#            print('#',i,'frame:',shared_state["frame"][i],':',shared_state["sleep"][i])
#            
#        for i in range(num_processes):
#            frame[i] = shared_state['frame'][i]
#        if max(frame) - min(frame) > 100:
#            idx = torch.LongTensor(frame).topk(num_processes-1)[1]
#            for i in range(num_processes):
#                if i in idx:
#                    shared_state['sleep'][i]=True
#                else:
#                    shared_state['sleep'][i]=False
                    
