import os
from typing import List
from enum import Enum
import typer
from loguru import logger
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import platform
from shutil import copy
import yaml
import json
import time
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from config import load_config_files
from data.augment import parse_augment_config
from data.dataset import LVNCDataset, get_classes_proportion
from data.maps_utils import compute_pta
from metrics.segmentation_callbacks import StoreTestSegmentationResults
from metrics.custom_loggers import CsvLogger
from utils import intercept_logger, set_global_seed, init_weights, update_dict_recursive


app = typer.Typer()


@app.command()
@logger.catch
@intercept_logger(logger=logger)
def run(train_cfg: List[Path] = typer.Option(...,
                                    help="Configuration files that will be used to obtain the final configuration",
                                    exists=True,
                                    readable=True)):

    original_cfg_dict, cfg = load_config_files(train_cfg)

    loss_config_fold = loss_config.copy()  
    net_model = cfg["network"]["model"]
    net_init = cfg["network"]["init"]
    net_config = cfg["network"]["configuration"]
    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]
    network = net_model(net_config, loss_config_fold, optim_config)

    model = net_model()
    model.load_weights("epoch=59-step=14759.ckpt")

    print(model.summary())

if __name__=="__main__":
    app()