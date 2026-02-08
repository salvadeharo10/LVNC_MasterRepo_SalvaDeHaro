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
    preproc_folder = "../LVNC_dataset/Preproc_2d_512_combinedresize"
    split_file = "../LVNC_dataset/split_5_cv_HCM_groupedby_patient.json"

    with open(split_file) as f:
            data = json.load(f)
            f.close()

    split1 = data["cross_validation"]

    for i in split1:
        split = i["train"]
        break

    net_model = cfg["network"]["model"]
    net_init = cfg["network"]["init"]
    net_config = cfg["network"]["configuration"]
    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]


    # Compute class weights if it is required
    loss_config_fold = loss_config.copy()        
    def parse_computed_weights(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if v=="COMPUTE_WEIGHTS":
                    weights = get_classes_proportion(LVNCDataset(preproc_folder, split["train"]), len(net_config["classes"]))
                    weights = 1/weights
                    weights = weights/sum(weights)
                    d[k] = weights
                else:
                    parse_computed_weights(v)
        elif isinstance(d, list):
            for l in d:
                parse_computed_weights(l)

    parse_computed_weights(loss_config_fold.get("params", {}))

    network = net_model(net_config, loss_config_fold, optim_config)
    init_weights(network, net_init)

    print(network)

if __name__=="__main__":
    app()
