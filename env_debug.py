#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:42:05 2024

@author: omega
"""

import time

def _game_loop(environment, render):
    """
    """
    obs = environment.reset()
    done = False

    if render:
        environment.render()
        time.sleep(0.5)
    iteration=0
    while not done:
        print('iter:',iteration)
        
        actions={}
        for ag in environment.agents_id:
            actions[ag]=environment.action_space.sample()
        
        # actions = environment.action_space.sample()
        # actions={'p0':actions}
        print('actions:', actions)
        nobs, nreward, ndone, _ = environment.step(actions)
        r=[]
        for key in nreward:
            r.append(nreward[key])
        
        if sum(r) > 0:
            print('reward:', nreward)

        if render:
            environment.render()
            time.sleep(0.5)

        # done = np.all(ndone)
        done=ndone['__all__']
        iteration+=1
    # print(env.players[0].score, env.players[1].score)
    
def main_loop(environment,game_count=10, render=False):
    # env = gym.make("Foraging-8x8-1p-1f-v2")
    obs = environment.reset()
    
    for episode in range(game_count):
        _game_loop(environment, render)