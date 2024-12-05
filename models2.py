#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:31:46 2022

@author: omega
"""

from gymnasium.spaces import Dict
# from gym.spaces import Dict #May raise problems because of the transition from gym to gymnasium in RLlib


from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()



#%% CC Action Mask Model
class CCActionMaskModel(TFModelV2):
    """Multi-agent model that implements a centralized value function.
    with an action mask model"""

    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name, 
                 **kwargs):

        orig_space = getattr(obs_space, "original_space", obs_space)
        
        # print(orig_space)
        # print(type(orig_space))
        
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces)
        
        
        
        # super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        super(CCActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)
        
        
        # import pdb
        # pdb.pdb.set_trace()
        # n_agents=model_config['env_config']['num_agents']
        # n_agents_other=n_agents-1 #NUmber of other agents
        
        
            
        
        
        # n_opp_agents=2
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        # obs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")
        # opp_obs = tf.keras.layers.Input(shape=obs_space.shape, name="opp_obs")
        # opp_act = tf.keras.layers.Input(shape=(2,), name="opp_act") #twostep game hs the same action space as flexenv mas environment
        # import pdb
        # pdb.pdb.set_trace()
        
        #%% Need to hardcode n_opp_agents everytime we change the number of agents
        n_opp_agents=2
        
        
        # h_size=95
        obs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")
        opp_obs = tf.keras.layers.Input(shape=(obs_space.shape[0]*n_opp_agents,), name="opp_obs")
        opp_act = tf.keras.layers.Input(shape=(n_opp_agents,), name="opp_act") #twostep game hs the same action space as flexenv mas environment
        

        # opp_act = tf.keras.layers.Input(shape=action_space.shape, name="opp_act") #twostep game hs the same action space as flexenv mas environment
        # import pdb
        # pdb.pdb.set_trace()
        concat_obs = tf.keras.layers.Concatenate(axis=1)([obs, opp_obs, opp_act])
        
        #BUG
        # Do we need the number of agents?
        central_vf_dense = tf.keras.layers.Dense(256, activation=tf.nn.tanh, name="c_vf_dense")(concat_obs)
        hidden_layer1 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(central_vf_dense)
        hidden_layer2 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(hidden_layer2)
        central_vf_out = tf.keras.layers.Dense(1, activation=None, name="c_vf_out")(hidden_layer3)
        
        # self.central_vf = tf.keras.Model(
        #     inputs=[obs, opp_obs, opp_act], outputs=central_vf_out,
        # name='c_vf_model')
        
        self.central_vf = tf.keras.Model(
            inputs=[obs, opp_obs, opp_act], outputs=central_vf_out,
        name='c_vf_model')


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state
        

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        # print('XXobs', obs)
        # print('XXop_obs', opponent_obs)
        # print('XXops_act',opponent_actions)
        
        

        # return tf.reshape(self.central_vf([obs, 
        #                                    opponent_obs, 
        #                                    tf.one_hot(tf.cast(opponent_actions,tf.int32), 4)]),[-1],)
        
        # input_tensor=[obs,opponent_obs,tf.one_hot(tf.cast(opponent_actions,tf.int32), 2)]
        input_tensor=[obs,opponent_obs,opponent_actions]
        
        
        # input_tensor=tf.keras.layers.Concatenate(axis=1)([obs, opponent_obs, tf.one_hot(tf.cast(opponent_actions,tf.int32), 4)])
            # print('XXobs', obs)])
            
 
        val=self.central_vf(input_tensor)
        
        # import pdb
        # pdb.pdb.set_trace() 
        return tf.reshape(val,[-1],)

    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()  # not used






#%% Action Mask Model
class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()