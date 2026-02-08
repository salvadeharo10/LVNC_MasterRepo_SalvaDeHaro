from curses import flash
import os
import glob
from pathlib import Path
from shutil import copy
from typing import List
import json, yaml
import pickle
import time

from loguru import logger
import typer

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from config import load_config_files
from data.dataset import LVNCDataset
from metrics.segmentation_callbacks import StoreTestSegmentationResults
from modules.transUnet import TransUNet, TransUNet2DEnsemble
from utils import intercept_logger
from sklearn.metrics import confusion_matrix

app = typer.Typer()

@app.command()
@intercept_logger(logger=logger)
@logger.catch
def run(train_cfg: Path = typer.Option(...,
                                    help="Configuration file in pickle format",
                                    exists=True,
                                    readable=True),
        split_file: Path = typer.Option(...,
                                    help="File containing dataset split configurarion",
                                    exists=True,
                                    readable=True),
        checkpoint_path: Path = typer.Option(...,
                                    help="Path to the checkpoint containing the weights that must be used during test",
                                    exists=True,
                                    readable=True),
        results_dir: Path = typer.Option(...,
                                    help="Output folder",
                                    exists=False,
                                    dir_okay=True),
        copy_checkpoint: bool = typer.Option(True,
                                    help="Whether or not checkpoint should be copied to results_dir"),
        copy_cfg_files: bool = typer.Option(False,
                                    help="Whether or not configuration files should be copied to results_dir")
        ):
    logger.add(os.path.join(results_dir, "output.log"), enqueue=True) 
    cfg = pickle.load(open(train_cfg, "rb"))

    preproc_folder = os.path.join(cfg["data_folder"], cfg["preproc_name"])

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if copy_cfg_files:
        copy(split_file, results_dir)
        copy(train_cfg, results_dir)

    if copy_checkpoint:
        copy(checkpoint_path, results_dir)

    # Read file with info regarding data splits
    split_info = json.load(open(split_file, "r"))
    test_split = split_info["test"]

    # Load network configuration
    net_config = cfg
    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]

    test_ds = LVNCDataset(preproc_folder, test_split)
    test_cfg_dl = cfg["data_loader"].get("test", cfg["data_loader"]["val"])
    test_dl = DataLoader(test_ds, **test_cfg_dl)

    # Initlize network
    logger.info(f"Initializing neural network...")
    network = net_model.load_from_checkpoint(checkpoint_path, net_config=net_config, loss_config=loss_config, optim_config=optim_config)

    # Callbacks
    test_segment_results_cb =  StoreTestSegmentationResults(
            output_folder=os.path.join(results_dir, "output"),
            classes= net_config["classes"],
            test_slice_list= test_split,
            raw_folder=os.path.join(cfg["data_folder"], "raw_dataset"),
            **cfg["callbacks"]["test_segmentation_results"]["params"]
        ) 

    trainer = Trainer(callbacks=[test_segment_results_cb], logger=None, default_root_dir=results_dir, checkpoint_callback=False, **cfg["trainer"])
    logger.info("Test begins...")
    trainer.test(network, test_dataloaders=test_dl)
    logger.success("Test has finished!")

    # log results
    logger.info("Test results:")
    logger.info(json.dumps(test_segment_results_cb.summary_results, indent=2, default=str))


@app.command()
@intercept_logger(logger=logger)
@logger.catch
def test_ensemble(experiment_dir: Path = typer.Option(..., exists=True, readable=True)):
    results_dir = os.path.join(experiment_dir, "test")
    train_cfg = os.path.join(experiment_dir, "CV/config.pick")
    cfg = pickle.load(open(train_cfg, "rb"))

    split_file = glob.glob(os.path.join(experiment_dir, "CV/split*.json"))[0]
    checkpoint_paths = []
    for fold_dir in glob.glob(os.path.join(experiment_dir, "Fold*")):
        checkpoint_paths.append(glob.glob(os.path.join(fold_dir, "epoch*.ckpt"))[0])

    preproc_folder = os.path.join(cfg["data_folder"], cfg["preproc_name"])
        
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    split_info = json.load(open(split_file, "r"))
    test_split = split_info["test"]

    # Load network configuration
    net_config = cfg

    img_size = net_config['transformer']["img_size"]
    patches_size = net_config['transformer']["patch_size"]
    net_config['transformer']["patches"]["grid"] = (int(img_size / patches_size), int(img_size / patches_size))

    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]

    test_ds = LVNCDataset(preproc_folder, test_split)
    test_cfg_dl = cfg["data_loader"].get("test", cfg["data_loader"]["val"])
    test_dl = DataLoader(test_ds, **test_cfg_dl)

    # Initlize network
    logger.info(f"Initializing neural network...")
    network = TransUNet2DEnsemble(checkpoint_paths, config=net_config, loss_config=loss_config, optim_config=optim_config)

    # Callbacks
    test_segment_results_cb =  StoreTestSegmentationResults(
            output_folder=os.path.join(results_dir, "output"),
            classes= net_config["unet"]["classes"],
            test_slice_list= test_split,
            output_size=(800, 800),
            raw_folder=os.path.join(cfg["data_folder"], "new_raw_dataset"),
            preproc_folder=os.path.join(cfg["data_folder"], cfg["preproc_name"]),
            **cfg["callbacks"]["test_segmentation_results"]["params"],
            compute_volume_diff=True
        ) 

    trainer = Trainer(callbacks=[test_segment_results_cb], logger=None, default_root_dir=results_dir, checkpoint_callback=False, **cfg["trainer"])
    logger.info("Test begins...")
    start = time.time()
    trainer.test(network, test_dataloaders=test_dl)
    end = time.time()
    tiempoTest = end-start
    logger.success("Test has finished!")
    # log results
    logger.info("Test results:")
    #Muestra todos los cortes del conjunto Test
    #logger.info(json.dumps(test_segment_results_cb.final_test_metrics, indent=2, default=str))
    #Solo aparece la media de los conjuntos HCM, X Y Hebron del conjunto Test
    #logger.info(json.dumps(test_segment_results_cb.summary_results, indent=2, default=str))
    tiempo = (time.strftime("%H:%M:%S",time.gmtime(tiempoTest)))
    
    pacientes = [] #para almacenar los id de pacientes del conjunto de test
    for corte in test_segment_results_cb.test_slice_list:
        if corte[0] not in pacientes:
            pacientes.append(corte[0])

    #Calculamos la Matriz de confusion.
    with open(results_dir+"/output/_result_volumes.json") as f:
        data = json.load(f)
        f.close()

    y_pred = []
    y_true = []
    classes = []
    for paciente in data['per_patient']:
        y_pred.append(paciente['VT% computed'])
        y_true.append(paciente['VT%'])
        classes.append(paciente['patient'])

    y_pred27 = filtrar(y_pred,27.4)
    y_true27 = filtrar(y_true,27.4)

    y_pred25 = filtrar(y_pred,25)
    y_true25 = filtrar(y_true,25)

    pacientesFPyFN27 = obtenerFPyFN(y_true27,y_pred27,classes)
    pacientesFPyFN25 = obtenerFPyFN(y_true25,y_pred25,classes)

    cf_matrix27 = confusion_matrix(y_true27,y_pred27).ravel()
    cf_matrix25 = confusion_matrix(y_true25,y_pred25).ravel()

    #Creamos el fichero informacion con el tiempo total, nºpacientes, nºcortes y matrices.
    tiempo17Cortes = sacarTiempo15cortes(preproc_folder,cfg,results_dir,net_config,network)
    with open(os.path.join(results_dir, "informacion.txt"), "w") as f:
            f.write(f"Tiempo de ejecucion del test: {tiempo}\n")
            f.write(f"Tiempo de ejecucion del test para 17 cortes, 3 pacientes: {tiempo17Cortes}\n")
            f.write(f"Pacientes totales: {len(pacientes)}\n")
            f.write(f"Cortes totales: {len(test_segment_results_cb.test_slice_list)}\n")
            f.write(f'Para threshold 27,4 tenemos:\n\tPacientes sin enfermedad: {cf_matrix27[0]}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {cf_matrix27[1]} {pacientesFPyFN27[0]}\n\tPacientes con enfermedad pero diagnosticados sanos: {cf_matrix27[2]} {pacientesFPyFN27[1]}\n\tPacientes con enfermedad: {cf_matrix27[3]}\n')
            f.write(f'Para threshold 25 tenemos:\n\tPacientes sin enfermedad: {cf_matrix25[0]}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {cf_matrix25[1]} {pacientesFPyFN25[0]}\n\tPacientes con enfermedad pero diagnosticados sanos: {cf_matrix25[2]} {pacientesFPyFN25[1]}\n\tPacientes con enfermedad: {cf_matrix25[3]}')
            f.close()

def filtrar(lista, threshold):
    listaFiltrada = []
    for i in lista:
        if(float(i) < threshold):
            listaFiltrada.append(0)
        else:
            listaFiltrada.append(1)
    return listaFiltrada
    
def obtenerFPyFN(y_true,y_pred,pacientes):
    resultado = []
    falsoPositivo = []
    falsoNegativo = []
    for i in range(len(y_true)):
        if(y_true[i] != y_pred[i]):
            if (y_true[i] == 0):
                falsoPositivo.append(pacientes[i])
            if (y_true[i] == 1):
                falsoNegativo.append(pacientes[i])
    resultado.append(falsoPositivo)
    resultado.append(falsoNegativo)
    return resultado

def sacarTiempo15cortes(preproc_folder,cfg,results_dir,net_config,network):
    test_split = [["MJGV", 7], ["MJGV", 8], ["MJGV", 9], ["P273", 4], ["P273", 5], ["P273", 6], ["P273", 7], ["P273", 8], ["X64", 2], ["X64", 3], ["X64", 4], ["X64", 5], ["X64", 6], ["X64", 7], ["X64", 8]]
    test_ds = LVNCDataset(preproc_folder, test_split)
    test_cfg_dl = cfg["data_loader"].get("test", cfg["data_loader"]["val"])
    test_dl = DataLoader(test_ds, **test_cfg_dl)

     # Callbacks
    test_segment_results_cb =  StoreTestSegmentationResults(
            output_folder=os.path.join(results_dir, "output"),
            classes= net_config["unet"]["classes"],
            test_slice_list= test_split,
            output_size=(800, 800),
            raw_folder=os.path.join(cfg["data_folder"], "new_raw_dataset"),
            preproc_folder=os.path.join(cfg["data_folder"], cfg["preproc_name"]),
            **cfg["callbacks"]["test_segmentation_results"]["params"],
        ) 

    trainer = Trainer(callbacks=[test_segment_results_cb], logger=None, default_root_dir=results_dir, checkpoint_callback=False, **cfg["trainer"])
    start = time.time()
    trainer.test(network, test_dataloaders=test_dl)
    end = time.time()
    tiempoTest = end-start
    tiempo = (time.strftime("%H:%M:%S",time.gmtime(tiempoTest)))
    return tiempo


if __name__=="__main__":
    app()
