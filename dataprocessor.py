#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:31:38 2024

@author: omega
"""

import pandas as pd
import numpy as np
import yaml
from utilities import utilities
# from metrics import Metrics


class YAMLParser:
    """
    parsing YAML files 
    """        
    def load_yaml(self,file):
        "outputs a dict from the config files"
        with open(file, 'r') as f:
            data = yaml.load(f,Loader=yaml.FullLoader) 
        return data
    
    def write_yaml(self,file, key, value):
        with open(file, 'r') as f:
            data = yaml.load(f,Loader=yaml.FullLoader)
        
        data[key] = value
         
        with open(file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        