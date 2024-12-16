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
exp_name=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
configs=ConfigsParser(configs_folder, exp_name)

file_experiment, algo_config,env_config=configs.get_configs()
train=YAMLParser().load_yaml(file_experiment)['train']
test=YAMLParser().load_yaml(file_experiment)['test']
#%%
env_config=YAMLParser().load_yaml(env_config)
env_for=ForagingEnv_r(players=env_config['players'],
                  max_player_level=env_config['max_player_level'],
                  field_size=(env_config['field_size_x'],
                              env_config['field_size_y']),
                  max_food=env_config['max_food'],
                  sight=env_config['sight'],
                  max_episode_steps=env_config['max_episode_steps'],
                  force_coop=env_config['force_coop'],
                  normalize_reward=env_config['normalize_reward'],
                  grid_observation=env_config['grid_observation'],
                  penalty=env_config['penalty'])


env_c=MultiAgentEnvCompatibility(env_for)

def env_creator(config):
    # return NormalizeObs(menv_base)  # return an env instance
    return MultiAgentEnvCompatibility(env_for)

register_env("lb-for-mas", env_creator)
# register_env("lb-for-mas", env_for)


#%% run the environment with random actions

# main_loop(env_for,game_count=1, render=True)

#%%
# obs=env_for.reset()
# env_for.render()
# # actions=env_for.action_space.sample()
# actions={'p0':0,'p1':2}
# actions={'p0':0}
# print(actions)
# obs2=env_for.step(actions)
# env_for.render()
# env_for.close()
# # env.check_multiagent_environments(env_c)


#%%
if train:
    experiment=Experiment(env_for, file_experiment)
    config=experiment.make_algo_config(algo_config)
    config_tune=experiment.make_tune_config()
    config_run=experiment.make_run_config(raylog.as_posix())
    
    resources=experiment.get_resources()
    trainable_obj=Trainable(file_experiment)
    trainable_func=trainable_obj.trainable
    trainable_resources = tune.with_resources(trainable_func, resources)
    
    ray.init(_system_config={"local_fs_capacity_threshold": 0.99,
                             "object_spilling_config": json.dumps({"type": "filesystem",
                                                                   "params": {"directory_path":[experiment.config['spill_dir']],}},)},)
    
    tuner = tune.Tuner(
          trainable_resources,
          param_space=config,
          tune_config=config_tune,
          run_config=config_run)
    
    results=tuner.fit()



#%% Test
if test:
    test=ExperimentTest(env_c,
              exp_name, 
              raylog,
              file_experiment,
              trainable_resources)
    
    tester=test.get_tester(trainable_resources)
    
    def policy_mapping_fn(agent_id):
        'Policy mapping function'
        return 'pol_' + agent_id
    
    main_loop(1,env_for,tester,policy_mapping_fn)
#%%
# trainer=PPO(config)
# trainer.train()
# p1=trainer.get_policy('pol_p0')
# p1.model.base_model.summary()

#%%


        
        

