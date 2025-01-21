#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:56:12 2024

@author: omega
"""
import argparse
import logging
import random
import time
# import gym
import numpy as np
# import lbforaging

# from gym.envs.registration import register

from environment_rllib import ForagingEnv_r
# from lbforaging.foraging import *

import ray #ray2.0 implementation
from ray import tune, air
from ray.tune import TuneConfig

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.rllib.models import ModelCatalog
# from ray.rllib.policy import MultiActionDistribution

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config
from ray.rllib.utils.pre_checks import env



import os
from os import path
from pathlib import Path

from experiment import Experiment
# from experiment_test import SimpleTests
from trainable import Trainable
from dataprocessor import YAMLParser
from utilities import ConfigsParser
from experiment_test import ExperimentTest

from env_debug import main_loop

import json

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
configs_folder=cwd / 'configs'
algos_config = configs_folder / 'algos_configs'



#%% exp_name + get configs
train_exp=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
test_exp=YAMLParser().load_yaml(configs_folder / 'test_config.yaml')['exp_name']
configs_train=ConfigsParser(configs_folder, train_exp)
configs_test=ConfigsParser(configs_folder, test_exp)

train_file_experiment, algo_config,_=configs_train.get_configs()
test_file_experiment,_,test_env_config=configs_test.get_configs()

#%%
test=YAMLParser().load_yaml(test_file_experiment)['test']
debug=YAMLParser().load_yaml(test_file_experiment)['debug']
#%% Test

    
env_config=YAMLParser().load_yaml(test_env_config)
env_for = ForagingEnv_r( players=env_config['players'],
                        field_size=(env_config['field_size_x'], env_config['field_size_y']),
                        sight=env_config['sight'],
                        max_episode_steps=env_config['max_episode_steps'],
                        force_coop=env_config['force_coop'],
                        normalize_reward=env_config['normalize_reward'],
                        grid_observation=env_config['grid_observation'],
                        penalty=env_config['penalty'],
                        randomize=env_config['randomize'],
                        network_cost=env_config['network_cost'],
                        num_storage=env_config['num_storage'],
                        num_network=env_config['num_network'],
                        storage_level=env_config['storage_level'],
                        network_level=env_config['network_level'],
                        min_consumption=env_config['min_consumption'],                       
                    )

env_c=MultiAgentEnvCompatibility(env_for)

def env_creator(config):
    # return NormalizeObs(menv_base)  # return an env instance
    return MultiAgentEnvCompatibility(env_for)

register_env("lb-for-mas", env_creator)

#%%
if test:
    
    trainable_func=Trainable(train_file_experiment)._trainable
    
    test=ExperimentTest(env_for,
              train_exp, 
              raylog,
              train_file_experiment,
              trainable_func)
    
    tester=test.get_tester(trainable_func)
    
    def policy_mapping_fn(agent_id):
        'Policy mapping function'
        return 'pol_' + agent_id
    
    main_loop(1,env_for,tester,policy_mapping_fn)

#%%
if debug:
    main_loop(1,env_for,[],[])
