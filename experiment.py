#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:59:37 2024

@author: omega
"""

# from models2 import ActionMaskModel, CCActionMaskModel
from ray.rllib.models import ModelCatalog
# ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
# ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

# from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial
# from obs_wrapper import *

# from auxfunctions_shiftenv import *
from termcolor import colored

from dataprocessor import YAMLParser

# from models2 import ActionMaskModel, CCActionMaskModel
from ray.rllib.models import ModelCatalog
# ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
# ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

# from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air, train
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial
# from obs_wrapper import *

# from auxfunctions_shiftenv import *
from termcolor import colored

#%%

class Experiment():
    def __init__(self,env,config):
        self.env=env
        self.parser=YAMLParser()
        self.config=self.parser.load_yaml(config)
        
        self.exp_name=self.config['exp_name']
        self.parser=YAMLParser()
        



    def get_policies(self):
        
        """
        Pol_type argument determines if there will be shared policy or each agent will have its own policy
        
        """
        pol_type=self.config['pol_type']
        
        config_pol={}
        
        # import pdb
        # pdb.pdb.set_trace()
        # env_cls=self._get_class(self.config['environment_cls'])
        # if not isinstance(self.env, env_cls):
        #     self.env=self.env.env
        
        #which  policy
        if pol_type=='agent_pol':
        
            policies={'pol_'+aid:(None,
                                self.env.observation_space,
                                self.env.action_space,
                                config_pol,) for aid in self.env.agents_id }
            
            policy_function=self.policy_mapping_fn
            
            
        elif pol_type=='shared_pol':
    
            policies={'shared_pol': (None,self.env.observation_space,self.env.action_space,config_pol,)}
            
            policy_function=self.policy_mapping_fn_shared
         
        # print('Policy Type:', colored(pol_type,'red'))
            
        return policies, policy_function    
     
    @staticmethod
    def policy_mapping_fn(agent_id,episode, worker, **kwargs):
        'Policy mapping function'
        return 'pol_' + agent_id
    
    @staticmethod
    def policy_mapping_fn_shared(agent_id, episode, worker, **kwargs):
        'Policy mapping function with shared policy'
        return 'shared_pol' # parameter sharing must return the same policy for any agent

    
    
    
    def make_algo_config(self, config_file):          
        #Config
        new_configs=self.parser.load_yaml(config_file)
        config_algo = PPOConfig().update_from_dict(new_configs)
        # import pdb
        # pdb.pdb.set_trace()
        #updates for environemnt 
        config_algo.environment(observation_space=self.env.observation_space,
                           action_space=self.env.action_space)
        
        #multiagent
        policies, policy_function = self.get_policies()
        
        config_algo.multi_agent(policies=policies,
                      policy_mapping_fn=policy_function)
        
        return config_algo
    
    
    def make_tune_config(self):
        config_tune=TuneConfig(mode=self.config['mode'],
                               metric=self.config['metric'],)
        return config_tune
    
    def make_run_config(self, results_dir):
        conf=train.RunConfig(verbose=self.config['verbose'], 
                                  name=self.exp_name,
                                  local_dir=results_dir,
                                  storage_path=results_dir)

        return conf

    def _get_class(self, class_name):
        """Retrieve a class object by its name."""
        return globals().get(class_name)
    
    
    def get_resources(self):
        pass
    
    def get_resources(self):
        num_cpu=self.config['num_cpu']
        num_gpu=self.config['num_gpu']
        cpu_factor=self.config['cpu_factor']
        
        b={'CPU': num_cpu,'GPU': num_gpu}
        
        resources=tune.PlacementGroupFactory([{'CPU': 1.0}]+ [b]*cpu_factor)
        
        return resources
        
        # resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': num_cpu}] * 6)

        