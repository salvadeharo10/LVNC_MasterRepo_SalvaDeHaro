import collections.abc

import os
import yaml

import importlib

import torch

from utils import update_dict_recursive

CONFIG_FILES_DIRECTORY = "./config_files"



def parse_config_dict(config):
    n_config = {}
    for k in config:
        if isinstance(config[k], collections.abc.Mapping):
            n_config[k] = parse_config_dict(config[k])
        elif isinstance(k, str) and k[:2]=="__":
            new_k = k[2:]
            try:
                n_config[new_k] = eval(config[k])
            except NameError as e:            
                module = importlib.import_module(config[k].rsplit('.', 1)[0])
                class_name = config[k].split(".")[-1]
                n_config[new_k] = getattr(module, class_name)
        elif isinstance(config[k], list):
            n_config[k] = [parse_config_dict(i) if isinstance(i, collections.abc.Mapping) else i for i in config[k]]
        else:
            n_config[k] = config[k]
    return n_config

def populate_config_dict(config):
    n_config = {}
    for k in config:
        if isinstance(config[k], collections.abc.Mapping):
            n_config[k] = populate_config_dict(config[k])
        elif isinstance(k, str) and k[0]=="_" and k[1]!="_":
            with open(os.path.join(CONFIG_FILES_DIRECTORY, config[k])) as f:
                n_config.update(yaml.safe_load(f))
        else:
            n_config[k] = config[k]
    return n_config

def load_config_files(config_files, recursive=False):
    config = {}
    for config_file in config_files:
        with open(config_file) as f:
            if recursive:
                config = update_dict_recursive(config, yaml.safe_load(f))
            else:
                config.update(yaml.safe_load(f))
    config = populate_config_dict(config)
    return config, parse_config_dict(config)
