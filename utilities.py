#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:59:38 2024

@author: omega
"""
import os
import inspect
from pathlib import Path
from termcolor import colored



class utilities:
    @staticmethod
    def print_info(message):
        frame = inspect.currentframe().f_back
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        print(colored(message,'red'))
        print(colored(f"@ '{function_name}' at line {line_number}",'green'))
        

class ConfigsParser():
    def __init__(self,configs_folder, exp_name):
        self.folder=configs_folder
        self.exp_name=exp_name
        self.make_configs()
        print(colored(f'Experiment Name ---> {self.exp_name}','red'))
        
    def traverse_folder(self):
        result = {}
        for root, dirs, files in os.walk(self.folder):
            # Initialize nested dictionary for subfolder
            current_dict = result
            for subdir in os.path.relpath(root, self.folder).split(os.path.sep):
                if subdir not in current_dict:
                    current_dict[subdir] = {}
                current_dict = current_dict[subdir]
            # Add files to subfolder dictionary
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                current_dict[file_name] = file
        return result
     
    
    def make_configs(self):
        """
        Output order
        - agents_config, 
        - apps_config
        - scene_config
        - problem_config
        - vars
        - experiment_config
        - self.algo_config_file
        
        exp_name must be a folder name or wwill return an error
        """
        from dataprocessor import YAMLParser #import here due to circular error

        files=self.traverse_folder()
        files=files[self.exp_name]
        self.exp_folder=Path(self.folder) / self.exp_name
        
        self.experiment_config=self.exp_folder / files['experiment_config']
        algo_config_file=YAMLParser().load_yaml(self.experiment_config)['algorithm']['config']
        self.algo_config_file=self.exp_folder /'algos_configs' / algo_config_file

        
        
    def get_configs(self):
        return self.experiment_config, self.algo_config_file
    
    def print_experiment_info(self):
        from dataprocessor import YAMLParser #import here due to circular error
        msg=YAMLParser().load_yaml(self.experiment_config)['info']
        print(colored(msg,'red'))
        
        

class FolderUtils():
    @staticmethod
    def get_file_in_folder(folder_path, file_type):
        """
        Scan a folder and return a list of CSV files in it.
        
        Args:
        - folder_path (str): The path to the folder to scan.
        
        Returns:
        - csv_files (list): A list of CSV files found in the folder.
        """
        csv_files = []
        for file in os.listdir(folder_path):
            if file.endswith(file_type):
                csv_files.append(os.path.join(folder_path, file))
        return csv_files
    
    @staticmethod
    def make_folder(folder):
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.makedirs(folder)
            print(colored('folder created' ,'red'),folder)



# parser=YAMLParser()
# parser.write_yaml(file_experiment,'exp_name','cenas')

# import os






        
    





    
 