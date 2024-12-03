# this should be a class!!!


from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config
import pandas as pd
import numpy as np
import random
from ray import tune
from auxfunctions_CC import CentralizedCritic
from termcolor import colored
from ray import train
from dataprocessor import YAMLParser

# import cProfile

class Trainable():
    def __init__(self,
             experiment_config):
        
        # self.parser=
        # self.algo_config=self.parser.load_yaml(algo_config)
        self.exper_config=YAMLParser().load_yaml(experiment_config)
        
        self.n_iters=self.exper_config['n_iters']
        self.checkpoint_freq=self.exper_config['checkpoint_freq']
    
    
    
    def _trainable(self, config):
        return self.trainable(config)
    
    def trainable(self,config):
        
        # n_iters=3
        # checkpoint_freq=1000
        
        # 
        algo_name=self.exper_config['algorithm']['name']
        
        if algo_name == 'PPO':
            self.trainer=PPO(config, env=config["env"])
        
        elif algo_name == 'CentralizedCritic':
            self.trainer=CentralizedCritic(config)
        
        print(colored(f'{algo_name} Trainer created.','red'))
                
        
        weights={}
        
        #set the seed
        seed=config['seed']
        np.random.seed(seed)
        random.seed(seed)    
        
        for i in range(self.n_iters):
            print(colored('training...','red'))
            train_results=self.trainer.train()
    
            #Metrics we are gonna log from full train_results dict
            metrics={'episode_reward_max', 
                     'episode_reward_mean',
                     'episode_reward_min',
                     'info', 
                     'episodes_total',
                     'agent_timesteps_total',
                     'training_iteration'}
            
            logs={k: train_results[k] for k in metrics}
                        
            results={**logs}
            # print(colored('Reporting...','green'))
            # train.report(results)
            # train.report(results, checkpoint=save_result.checkpoint)
            if i % self.checkpoint_freq == 0:
                print(colored('Iteration:','red'),i)
                print(colored('Reporting and checkpointing...','green'))
                save_result=self.trainer.save()
                # trainer.save_checkpoint()
                cp_path=save_result.checkpoint.path
                checkpoint_object=train.Checkpoint.from_directory(cp_path)
                train.report(results, checkpoint=checkpoint_object)
            else:
                print(colored('Reporting...','green'))
                train.report(results)
                
            # session.report
            

            
        self.trainer.stop()


# def trainable_mas(config):
    
#     n_iters=3
#     checkpoint_freq=1000
    
#     # trainer=PPO(config, env=config["env"])
#     trainer=CentralizedCritic(config)
#     print(colored('Trainer created...','red'))
    
#     # train.get_checkpoint(args, kwargs)
    
    
#     weights={}
    
#     #set the seed
#     seed=config['seed']
#     np.random.seed(seed)
#     random.seed(seed)    
    
#     for i in range(n_iters):
#         print(colored('training...','red'))
#         train_results=trainer.train()
#         # train_results=cProfile.run('trainer.train()')
# #       
#         #Metrics we are gonna log from full train_results dict
#         metrics={'episode_reward_max', 
#                  'episode_reward_mean',
#                  'episode_reward_min',
#                  'info', 
#                  'episodes_total',
#                  'agent_timesteps_total',
#                  'training_iteration'}
        
#         logs={k: train_results[k] for k in metrics}
        
#         #get model weights
#         # for k, v in trainer.get_policy().get_weights().items():
#         #             weights["FCC/{}".format(k)] = v
        
#         # save checkpoint every checkpoint_freq
#         # if i % checkpoint_freq == 0: 
#         #     print(colored('Saving checkpoint...','green'))
#             # checkpoint=trainer.save(tune.get_trial_dir())
#             # checkpoint_dir=train.get_context().get_trial_dir()
  
#         # print(colored('Saving checkpoint...','green'))  
#         # checkpoint_dir=train.get_context().get_trial_dir()
#         # print(colored('checkpoint dir','red'), checkpoint_dir)  
#         # checkpoint=trainer.save(checkpoint_dir)
        
        
        
        
#         # checkpoint=trainer.save(tune.get_trial_dir())
#         # print(colored('Checkpoint saved...','green'))
#         #evaluate agent
#         # print('evaluating...')
#         # eval_results=trainer.evaluate()
#         # eval_metrics={'episode_reward_max', 
#         #       'episode_reward_mean',
#         #       'episode_reward_min',}
#         # eval_logs={'evaluation':{}}
#         # eval_logs['evaluation']={k: eval_results['evaluation'][k] for k in eval_metrics}
        
#         # results={**logs,**weights,**eval_logs}
#         # print(colored('Results...','green'))

#         #save the checkpoint
#         # save_result=trainer.save()
#         # cp_path=save_result.checkpoint.path
#         # save_result
#         # checkpoint=trainer.save(train.get_context().get_trial_dir())
#         # print(colored('Saving checkpoint...','green'))
#         # print('checkpoint', train.get_context().get_trial_dir())

#         # checkpoint_dir=train.get_context().get_trial_dir()
#         # print(colored('checkpoint dir','red'), checkpoint_dir)  
        
#         # print(checkpoint)   
        
#         # #Create checkpoint from directory
#         # save_result=trainer.save()
#         # cp_path=save_result.checkpoint.path
#         # checkpoint_object=train.Checkpoint.from_directory(cp_path)
        
        
#         # print(colored('checkpoint get trial','red'), train.get_context().get_trial_dir())  
#         # print(colored('checkpoint object path','red'), checkpoint_object.path)  
            
        
#         # save_result=trainer.save(train.get_context().get_trial_dir())
#         # print(colored('checkpoint from save_result','red'), save_result.checkpoint)  
        
#         results={**logs}
#         # print(colored('Reporting...','green'))
#         # train.report(results)
#         # train.report(results, checkpoint=save_result.checkpoint)
#         if i % checkpoint_freq == 0:
#             print(colored('Iteration:','red'),i)
#             print(colored('Reporting and checkpointing...','green'))
#             save_result=trainer.save()
#             # trainer.save_checkpoint()
#             cp_path=save_result.checkpoint.path
#             checkpoint_object=train.Checkpoint.from_directory(cp_path)
#             train.report(results, checkpoint=checkpoint_object)
#         else:
#             print(colored('Reporting...','green'))
#             train.report(results)
            
#         # session.report
        

        
#     trainer.stop()






# n_iters=200
# checkpoint_freq=10

# def trainable(config):
    
#     trainer=PPO(config, env=config["env"])
#     weights={}
    
#     #set the seed
#     seed=config['seed']
#     np.random.seed(seed)
#     random.seed(seed)    
    
#     for i in range(n_iters):
#         print('training...')
#         train_results=trainer.train()
# # 
#         #Metrics we are gonna log from full train_results dict
#         metrics={'episode_reward_max', 
#               'episode_reward_mean',
#               'episode_reward_min',
#               'info', 
#               'episodes_total',
#               'agent_timesteps_total',
#               'training_iteration'}
        
#         logs={k: train_results[k] for k in metrics}
        
#         #get model weights
#         for k, v in trainer.get_policy().get_weights().items():
#                     weights["FCC/{}".format(k)] = v

#         #save checkpoint every checkpoint_freq
#         if i % checkpoint_freq == 0: 
#             checkpoint=trainer.save(tune.get_trial_dir())
        
#         #evaluate agent
#         print('evaluating...')
#         # eval_results=trainer.evaluate()
#         eval_metrics={'episode_reward_max', 
#               'episode_reward_mean',
#               'episode_reward_min',}
#         eval_logs={'evaluation':{}}
#         # eval_logs['evaluation']={k: eval_results['evaluation'][k] for k in eval_metrics}
        
#         results={**logs,**weights,**eval_logs}
#         # results={**eval_logs}
#         tune.report(results)
        
#     trainer.stop()





