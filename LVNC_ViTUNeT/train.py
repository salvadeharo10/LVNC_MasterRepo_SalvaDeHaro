import sys
# Anadir ambas rutas posibles
#sys.path.append("/home/salvador/.local/lib/python3.8/site-packages/")
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
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.profilers import PyTorchProfiler

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.profilers

from config import load_config_files
from data.augment import parse_augment_config
from data.dataset import LVNCDataset, get_classes_proportion
from metrics.segmentation_callbacks import StoreTestSegmentationResults
from metrics.custom_loggers import CsvLogger
from utils import intercept_logger, init_weights, set_global_seed, update_dict_recursive

from modules.transUnet import  TransUNet
import psutil
import GPUtil
import csv


# Callback personalizado para registrar el uso de recursos durante el entrenamiento
class ResourceLoggerCallback(LambdaCallback):
    def __init__(self, logfile="resources.csv"):
        super().__init__()
        self.logfile = logfile
        # Inicializa el archivo CSV con encabezados
        with open(self.logfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "cpu%", "ram_MB", "gpu%", "gpu_mem_MB"])

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        # Obtiene metricas del sistema
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / 1024**2  # en MB
        gpus = GPUtil.getGPUs()
        gpu = gpus[0].load * 100 if gpus else 0
        gpu_mem = gpus[0].memoryUsed if gpus else 0
        epoch = trainer.current_epoch

        # Guarda en CSV
        with open(self.logfile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, cpu, ram, gpu, gpu_mem])

        # Guarda en TensorBoard (si logger tiene experiment)
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and logger.experiment is not None:
                logger.experiment.add_scalar("GPU/usage", gpu, epoch)
                logger.experiment.add_scalar("GPU/memory_MB", gpu_mem, epoch)
                logger.experiment.add_scalar("CPU/usage", cpu, epoch)
                logger.experiment.add_scalar("RAM/used_MB", ram, epoch)


            
app = typer.Typer()


@app.command()
@logger.catch
@intercept_logger(logger=logger)
def run(train_cfg: List[Path] = typer.Option(...,
                                    help="Configuration files that will be used to obtain the final configuration",
                                    exists=True,
                                    readable=True),
        split_file: Path = typer.Option(...,
                                    help="File containing dataset split configurarion",
                                    exists=True,
                                    readable=True),
        augmentation_file: Path = typer.Option(None,
                                    help="File defining the augmentation steps applied to the train set",
                                    exists=True,
                                    readable=True),
        results_dir: Path = typer.Option(None,
                                    help="Output directory",
                                    exists=True,
                                    readable=True,
                                    writable=True),
        folds: List[int] = None):

    original_cfg_dict, cfg = load_config_files(train_cfg)
    
    # Define experiment name based on datetime
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y%m%d%H%M%S")
    experiment_name = "{}_{}".format(platform.node(), current_datetime_str)

    if results_dir is None:
        # Define and create main results_dir if not defined
        results_dir = os.path.join("./logs", experiment_name)
        Path(results_dir).mkdir(parents=True, exist_ok=False)
    
    preproc_folder = os.path.join(cfg["data_folder"], cfg["preproc_name"])
    experiment_logger = logger.add(os.path.join(results_dir, "output.log"), enqueue=True, filter=lambda record: "fold" not in record["extra"])    
    logger.info(f"Preproc dataset: {os.path.basename(preproc_folder)} ; located in {os.path.realpath(preproc_folder)}")
    logger.info("Configuration: ")
    logger.info(original_cfg_dict)

    # Random seed
    seed = cfg.get("seed", None)
    if seed:
        set_global_seed(seed)

    def copy_config_files(dest_folder):
        # Copy config files to results dir
        copy(split_file, dest_folder)
        with open(os.path.join(dest_folder, "config.yaml"), 'w') as f:
            yaml.dump(original_cfg_dict, f, default_flow_style=False, sort_keys=False)
        with open(os.path.join(dest_folder, "config.pick"), 'wb') as f: # Pickle file with configuration file. Easier for evaluating test file and resuming training
            pickle.dump(cfg, f) 
        if augmentation_file:
            copy(augmentation_file, dest_folder)

    # Read file with info regarding data splits
    split_info = json.load(open(split_file, "r"))

    # Check if CV or single train    
    cross_val_split = split_info.get("cross_validation", None)
    train_split = split_info.get("train/val", None)
    assert (cross_val_split is None) != (train_split is None), "Only one of cross_validation or train/val info must be present in the split file"
    if cross_val_split is not None:
        if not folds:
            folds = range(len(cross_val_split))
        logger.info(f"Folds: {folds}")

        cv_folder = os.path.join(results_dir, "CV")
        Path(cv_folder).mkdir(parents=True, exist_ok=True)
        copy_config_files(cv_folder)
    else:
        val_split = split_info["val"]
        training_folder = os.path.join(results_dir, "training")
        Path(training_folder).mkdir(parents=True, exist_ok=True)
        
    
########## Load network configuration ###########################################################
    config_net = cfg #load configuracion
    
    img_size = config_net['transformer']["img_size"]
    patches_size = config_net['transformer']["patch_size"]
    config_net['transformer']["patches"]["grid"] = (int(img_size / patches_size), int(img_size / patches_size))
	
    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]
###################################################################################################	
   
    #cv_results = []
    tiempoTrain = 0
    for i in folds:
        logger.info(f"################# Fold {i} #################")
        # Initialize i-fold folder
        fold_dir = os.path.join(results_dir, f"Fold{i}")
        
        # Train-test split

        test_dir = os.path.join(fold_dir, "test")
        Path(test_dir).mkdir(parents=True, exist_ok=True)

        # Load augmentations
        transform = parse_augment_config(augmentation_file)

        # Create datasets
        split = cross_val_split[i]
        #get_classes_proportion(LVNCDataset(preproc_folder, split["train"]), num_classes=4, batch_size=32)
        train_split = split["train"]
        test_split = split["val"]
        train_split, val_split = train_test_split(train_split, test_size=0.2)
        train_ds = LVNCDataset(preproc_folder, train_split, transform)
        val_ds = LVNCDataset(preproc_folder, val_split)
        train_cfg_dl = cfg["data_loader"]["train"]
        val_cfg_dl = cfg["data_loader"]["val"]
        train_dl = DataLoader(train_ds, **train_cfg_dl)
        val_dl = DataLoader(val_ds, **val_cfg_dl)
        # Compute class weights if it is required
        loss_config_fold = loss_config.copy()        
        def parse_computed_weights(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if v=="COMPUTE_WEIGHTS":
                        weights = get_classes_proportion(LVNCDataset(preproc_folder, split["train"]), len(config_net['unet']["classes"]))
                        weights = 1/weights
                        weights = weights/sum(weights)
                        d[k] = weights
                    else:
                        parse_computed_weights(v)
            elif isinstance(d, list):
                for l in d:
                    parse_computed_weights(l)
        parse_computed_weights(loss_config_fold.get("params", {}))
        

        ############## Initlize network ##################################################################
        logger.info(f"Initializing neural network...")
        
        #creamos el modelo
        network = TransUNet(config_net, loss_config_fold, optim_config)

        #cargamos el modelo preentrenado con ImageNet21k
        init_weights(network.encoder.embeddings.hybrid_model, 'kaiming')
        network.load_from(weights=np.load(config_net['transformer']["pretrained_path"]))
        init_weights(network.decoder, 'kaiming')

        logger.success(f"Neural network ready!")
        logger.debug(network)
        ###################################################################################################	
        
        
        # Callbacks
        logger.info(f"Initializing callbacks...")
        callbacks = []
        # Callback to store the best checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            dirpath=fold_dir,
            mode='min',
            save_last=False,
            save_weights_only=True  # solo pesos del modelo
        )
        callbacks.append(checkpoint_callback)
        # Early stopping callback
        if cfg["callbacks"]["early_stopping"]["apply"]:
            callbacks.append(EarlyStopping(**cfg["callbacks"]["early_stopping"]["params"]))
        if cfg["callbacks"]["test_segmentation_results"]["apply"]:
            test_segment_results_cb =  StoreTestSegmentationResults(
                    output_folder=test_dir,
                    classes= config_net['unet']["classes"],
                    test_slice_list= split["val"],
                    raw_folder=os.path.join(cfg["data_folder"], "new_raw_dataset"),
                    preproc_folder=os.path.join(cfg["data_folder"], cfg["preproc_name"]),
                    **cfg["callbacks"]["test_segmentation_results"]["params"],
                    compute_volume_diff=True
                ) 
            callbacks.append(test_segment_results_cb)
            
        # Callback de registro de consumo de recursos
        callbacks.append(ResourceLoggerCallback(os.path.join(fold_dir, "resources.csv")))

        
        logger.success(f"Callbacks are ready!")

        # Loggers
        loggers = []
        if cfg["loggers"]["use_csv_logger"]:
            loggers.append(
                CsvLogger(
                    os.path.join(fold_dir, "train.csv"),
                    os.path.join(fold_dir, "val.csv"),
                    ["epoch", "train/loss"],
                    ["epoch", "val/loss", "val/pta_difference"] + [f"val/dice_{c}" for c in config_net['unet']["classes"]]
                )
            )
        if cfg["loggers"]["use_tensorboard"]:
            loggers.append(
                TensorBoardLogger(
                    "./logs",
                    experiment_name,
                    f"Fold{i}"
                )
            )

########### Initialize the trainer and fit ########################################################
        start = time.time()
      
        # Crea el profiler
        profiler = pytorch_lightning.profilers.PyTorchProfiler(
            dirpath=os.path.join(fold_dir, "profiler"),
            filename="perf",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=0
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(fold_dir, "profiler")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        
        # Ahora si creamos el Trainer CORRECTAMENTE
        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=fold_dir,
            profiler=profiler,  # aqui pasamos el profiler
            **cfg["trainer"]
        )
        
        trainer.fit(network, train_dl, val_dl)
        
        end = time.time()
        tiempoTrain = tiempoTrain + (end-start)
        logger.success("Training process is finished")
        logger.success(f"Best checkpoint stored in {checkpoint_callback.best_model_path} with a score of {checkpoint_callback.best_model_score}")
###################################################################################################
        
        
        # Evaluate on test set using the best weights
        test_ds = LVNCDataset(preproc_folder, test_split)
        test_dl = DataLoader(test_ds,  **val_cfg_dl)
        trainer.test(dataloaders=test_dl)

        #if cfg["callbacks"]["test_segmentation_results"]["apply"]:
        #    cv_results.append(test_segment_results_cb.mean_dice)
        logger.success(f"Fold {i} finished!")
        torch.cuda.empty_cache()


    # Now we aggregate the results of all the folds
    summary_file_path = os.path.join(results_dir, "summary.txt")
    final_train_losses = []
    test_results = []
    with open(summary_file_path, "w") as f:
        for i in folds:
            fold_dir = os.path.join(results_dir, f"Fold{i}")
            if cfg["loggers"]["use_csv_logger"]:            
                df_train = pd.read_csv(os.path.join(fold_dir, "train.csv"))
                df_val = pd.read_csv(os.path.join(fold_dir, "val.csv"))
                best_idx = df_val["val/loss"].argmin()
                train_loss_in_best_idx = df_train.iloc[best_idx]["train/loss"]
                final_train_losses.append(train_loss_in_best_idx)
                logger.info(f"Fold {i} achieved the lowest val/loss at epoch {best_idx} where the train/loss was {train_loss_in_best_idx} and the validation metrics were:\n{df_val.iloc[best_idx]}")
            if cfg["callbacks"]["test_segmentation_results"]["apply"]:
                d = json.load(open(os.path.join(fold_dir, "test", "_result.json"), "r"))["aggregation"]["mean"]
                if cfg["callbacks"]["test_segmentation_results"].get("params", {}).get("compute_volume_diff", False):
                    d_vol = json.load(open(os.path.join(fold_dir, "test", "_result_volumes.json"), "r"))["aggregation_vt_diff"]
                    d_vol = {s: {"VT% diff": d_vol[s]["mean"]} for s in d_vol}
                    d = update_dict_recursive(d, d_vol)
                if cfg["callbacks"]["test_segmentation_results"].get("params", {}).get("group_results_by", None):
                    test_results.append(pd.DataFrame.from_dict(d, orient="index"))
                else:
                    test_results.append(pd.DataFrame.from_dict(d)) # We only want the mean here
    
        logger.info("El tiempo de ejecucion del trainer es de: ")
        logger.success(time.strftime("%H:%M:%S",time.gmtime(tiempoTrain)))

        df_test_results = pd.concat(test_results).agg(["mean", "std"])
        df_test_results.rename(index={"VT% diff": "value"}, inplace=True)
        logger.info(f"Average CV results:\n{df_test_results}")



class StoppingCriterion(Enum):
    """
    loss: once the training loss is lower than the average training loss during cross-validation (at the epoch of minimum validation loss), the training process is stopped.
    epoch: once the number of the epoch is higher than the average epoch of the cross-validation in which the minimum validation loss was reached, the training process is stopped.
    """
    CV_loss = "loss" 
    CV_epoch = "epoch"


@app.command()
@logger.catch
@intercept_logger(logger=logger)
def train_after_cv(experiment_dir: Path = typer.Option(...,
                                        help="Directory of the experiment containing the CV folder",
                                        exists=True,
                                        readable=True,
                                        writable=True),
                    split_file: Path = typer.Option(None,
                                        help="Split file to use. If `None`, it will use the whole training of the cross validation as training (no validation)",
                                        exists=True,
                                        readable=True),
                    stopping_criterion: StoppingCriterion = typer.Option("loss", help="What metric of the cross-validation should be used to stop the training process.")):
    # We load the 
    train_cfg = os.path.join(experiment_dir, "config.pick")
    original_cfg_dict, cfg = load_config_files(train_cfg)
    # TODO

    

if __name__=="__main__":
    app()

