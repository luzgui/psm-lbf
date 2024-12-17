#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:21:34 2024

@author: omega
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:18:23 2023

@author: omega
"""

import gymnasium as gym

import ray #ray2.0 implementation
from ray import tune, air
from ray.tune import analysis, ExperimentAnalysis, TuneConfig
from ray.tune.experiment import trial

#PPO algorithm
from auxfunctions_CC import CentralizedCritic
from ray.rllib.algorithms.ppo import PPO, PPOConfig #trainer and config
from ray.rllib.env.env_context import EnvContext
#models
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.pre_checks import env

#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#System
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime

#Custom functions
# from shiftenvRLlib import ShiftEnv
# from auxfunctions_shiftenv import *
# from plotutils import *
# from models2 import ActionMaskModel, CCActionMaskModel



import random

from trainable import *
# from obs_wrapper import *

# from shiftenvRLlib_mas import ShiftEnvMas

# from auxfunctions_CC import *

# Custom Model
# ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
# ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

#profiling
import cProfile
import pstats
from pstats import SortKey
from icecream import ic

#pyomo
# from auxfunctions_opti import *
from pyomo.environ import *
from pyomo.opt import SolverFactory
import scipy.io as sio
import re 
from itertools import compress
from rich import print
from rich.console import Console
from rich.syntax import Syntax

import json

#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'


# from dataprocessor import DataPostProcessor, YAMLParser
from utilities import utilities

#%% Test Class
class ExperimentTest():
    def __init__(self,test_env, 
                 test_exp_name, 
                 log_dir, 
                 train_experiment_config_file,
                 trainable):
        
        self.env=test_env
        self.exp_name=test_exp_name
        self.dir=log_dir
        self.train_experiment_config=YAMLParser().load_yaml(train_experiment_config_file)
        self.trainable=trainable
        
        self.algo_name=self.train_experiment_config['algorithm']['name']
        
        if self.algo_name == 'PPO':
            self.tester=PPO
        
        elif self.algo_name == 'CentralizedCritic':
            self.tester=CentralizedCritic
        
        # self.tester=tester
        
    def simple_transition(env):
        pass
    
    
    def get_tester(self, trainable):

        experiment_path=os.path.join(self.dir, self.exp_name)
        # import pdb
        # pdb.pdb.set_trace()
        
        spill_1=raylog / 'spill1'
        spill_2=raylog / 'spill2'

        ray.init(_system_config={"local_fs_capacity_threshold": 0.99,
                                 "object_spilling_config": json.dumps({"type": "filesystem",
                                                                       "params": {"directory_path":[spill_1.as_posix(),
                                                                                                    spill_2.as_posix()],}},)},)
        
        restored_tuner = tune.Tuner.restore(experiment_path,trainable=self.trainable)
        # restored_tuner = tune.Tuner.restore(experiment_path,trainable=trainable, resume_unfinished=True)
        result_grid = restored_tuner.get_results()
        best_res=result_grid.get_best_result()
        config=best_res.config
        
        utilities.print_info('num_workers changed sue to resource scarcicity')
        config['num_workers']=1
        config['num_gpus']=0
        config['num_gpus_per_worker']=0
        checkpoint=best_res.checkpoint
        tester=self.tester(config, env=config["env"])
        tester.restore(checkpoint)
        print('restored the following checkpoint',checkpoint)
        
 
        # print(self.config['mode'],checkpoint)
        
        return tester

    

class SimpleTests:
    def __init__(self,test_env):
        self.env=test_env
        self.processor=DataPostProcessor()
        
    def get_action_plan(self):
        actions={}
        starts=dict(zip(self.env.agents_id, [0,10,30,32,36,45,50][0:len(self.env.agents_id)]))
        
        for ag in self.env.agents_id:
            agent=self.env.com.get_agent_obj(ag)
            D=agent.apps[0].duration/self.env.tstep_size
            actions[ag]=self.create_binary_vector(self.env.Tw,D,starts[ag])
        
        return actions
          
    def create_binary_vector(self, T, D, t):
        """
        Create a binary vector with zeros everywhere except for a specific duration D starting at time t.

        """
        binary_vector = np.zeros(T, dtype=int)
        # import pdb
        # pdb.pdb.set_trace()
        binary_vector[t:int(t+D)] = 1
        return binary_vector
            
        
    def transition_test(self, var_out):
        obs0=self.env.reset()
        action=self.env.action_space_sample(keys=None)
        obs1=self.env.step(action)
        
        if var_out=='state_hist':
            return self.env.state_hist
        elif var_out=='obs':
            print('action', action)
            return obs1
        
    
        
    def episode_test(self):
        
        obs=self.env.reset()
        action_plan=self.get_action_plan()
        actions={}
        
        for i in range(self.env.Tw):
            actions = {aid: action_plan[aid][i] for aid in self.env.agents_id}  
            print('iteration', i)
            obs, reward, done, info = self.env.step(actions)
        
            
        
        return self.env



    
    
