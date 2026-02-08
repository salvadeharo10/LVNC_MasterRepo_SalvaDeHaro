import random
import logging
import functools
from loguru import logger
import collections.abc

import torch
import torch.nn as nn
from torch.nn import init
from functools import reduce
import numpy as np

class InterceptHandler(logging.Handler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        logger_opt = self.logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelno, record.getMessage())

def intercept_logger(_func=None, *, logger):
    def decorator_intercept_logger(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            intercept_handler = InterceptHandler(logger)
            logging.basicConfig(handlers=[intercept_handler], level=0)
            logging.getLogger("pytorch_lightning").addHandler(intercept_handler)
            func(*args, **kwargs)
        return wrapper
    if _func is None:
        return decorator_intercept_logger
    else:
        return decorator_intercept_logger(_func)


# https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def pytorch_count_trainable_params(model):
  return sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())

def set_global_seed(seed: int, deterministic: bool =False, benchmark: bool = False):
    # https://pytorch.org/docs/stable/notes/randomness.html
    # https://pytorch.org/docs/1.7.0/notes/randomness.html
    logger.info(f"Seed: {seed}; Deterministic: {deterministic}; Benchmark: {benchmark}")
    torch.manual_seed(seed)  # Works for both CUDA and CPU
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = benchmark
    # Required for albumentations
    random.seed(seed)

# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def update_dict_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d