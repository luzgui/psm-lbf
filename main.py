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

file_experiment, algo_config=configs.get_configs()

#%%

env_for=ForagingEnv_r(players=2,
                  max_player_level=5,
                  field_size=(5,5),
                  max_food=5,
                  sight=2,
                  max_episode_steps=100,
                  force_coop=True,
                  normalize_reward=True,
                  grid_observation=False,
                  penalty=0.0)


env_c=MultiAgentEnvCompatibility(env_for)

def env_creator(config):
    # return NormalizeObs(menv_base)  # return an env instance
    return MultiAgentEnvCompatibility(env_for)

register_env("lb-for-mas", env_creator)


#%% run the environment with random actions

main_loop(env_for,game_count=1, render=True)

#%%
obs=env_for.reset()
# env_for.render()
# # actions=env_for.action_space.sample()
actions={'p0':0,'p1':2}
# actions={'p0':0}
print(actions)
obs2=env_for.step(actions)
# env_for.render()
# env_for.close()
# # env.check_multiagent_environments(env_c)


#%%
experiment=Experiment(env_for, file_experiment)
config=experiment.make_algo_config(algo_config)
config_tune=experiment.make_tune_config()
config_run=experiment.make_run_config(raylog.as_posix())

resources=experiment.get_resources()
trainable_obj=Trainable(file_experiment)
trainable_func=trainable_obj.trainable
trainable_resources = tune.with_resources(trainable_func, resources)

spill_1=raylog / 'spill1'
spill_2=raylog / 'spill2'

ray.init(_system_config={"local_fs_capacity_threshold": 0.99,
                         "object_spilling_config": json.dumps({"type": "filesystem",
                                                               "params": {"directory_path":[spill_1.as_posix(),
                                                                                            spill_2.as_posix()],}},)},)

tuner = tune.Tuner(
      trainable_resources,
      param_space=config,
      tune_config=config_tune,
      run_config=config_run)




results=tuner.fit()


#%%
# trainer=PPO(config)
# trainer.train()
# p1=trainer.get_policy('pol_p0')
# p1.model.base_model.summary()

#%%

        
        


# def get_env_name(s,f,p,c):
    
#     env_name="Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(s, p, f, "-coop" if c else "")

#     register(
#         id=env_name,
#         entry_point="lbforaging.foraging:ForagingEnv",
#         kwargs={
#             "players": p,
#             "max_player_level": 3,
#             "field_size": (s, s),
#             "max_food": f,
#             "sight": s,
#             "max_episode_steps": 50,
#             "force_coop": c,
#         },
#     )
    
#     print('created and registered environment: ', env_name)
#     return env_name

# #%%

# env = gym.make(get_env_name(s=6, p=2, f=2, c=True))
# main(env,10,render=True)

# k=0
# while k < 15:
#     print(env.action_space.sample())
#     k+=1


    
# #%%

# env = gym.make(get_env_name(s=6, p=2, f=2, c=True))
# env.reset()
# env.render()
# actions=env.action_space.sample()
# print(actions)
# env.step(actions)
# env.render()
# env.close()
    
    

        