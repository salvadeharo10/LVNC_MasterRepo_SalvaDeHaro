import os
import glob
import json
import pickle
import time
import csv
import psutil
import pynvml

import pandas as pd
from data.maps_utils import logits_to_onehot  # Asumiendo que existe esta funcion en tu repo


from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from config import load_config_files
from data.dataset import LVNCDataset
from metrics.segmentation_callbacks import StoreTestSegmentationResults
from modules.transUnet import TransUNet2DEnsemble
from loguru import logger
import typer

app = typer.Typer()



def compute_dice(output: torch.Tensor, target: torch.Tensor, num_classes: int, compute_classes=None, smooth=1e-8):
    '''
        Funcion para calcular los coeficientes Dice sobre los cortes de Test
    '''
    if compute_classes is None:
        compute_classes = list(range(num_classes))
    output = output[:, compute_classes, :, :]
    target = target[:, compute_classes, :, :]
    intersection = output * target
    dice = (2 * torch.sum(intersection, dim=(2, 3)) + smooth) / (
        torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) + smooth)
    return dice




@app.command("run")
def test_ensemble_with_profiling(
    experiment_dir: Path = typer.Option(..., exists=True, readable=True),
    device: str = typer.Option("auto", help="Device to use: 'cpu', 'gpu', or 'auto'"),
    precision: str = typer.Option("32", help="Precision mode: '32', '16-mixed', 'bf16-mixed'"),
    batch_size: int = typer.Option(None, help="Batch size to override the default")
):
    # --- Select Device ---------------------------------------------
    if device == "cpu":
        selected_device = torch.device("cpu")
    elif device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            selected_device = torch.device("cuda:0")
        else:
            logger.error("CUDA requested but not available. Falling back to CPU.")
            selected_device = torch.device("cpu")
    elif device == "auto":
        selected_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        logger.error(f"Unknown device option: {device}. Use 'cpu', 'gpu', or 'auto'.")
        raise typer.Exit(code=1)

    logger.info(f"Using device: {selected_device}")
    logger.info(f"Using precision mode: {precision}")

    # --- Setup -----------------------------------------------------
    results_dir = os.path.join(experiment_dir, f"test_Precision{precision}_BS{batch_size}_{device}")
    train_cfg = os.path.join(experiment_dir, "CV/config.pick")
    cfg = pickle.load(open(train_cfg, "rb"))

    split_file = glob.glob(os.path.join(experiment_dir, "CV", "*split*.json"))[0]
    checkpoint_paths = [glob.glob(os.path.join(fold_dir, "epoch*.ckpt"))[0]
                        for fold_dir in glob.glob(os.path.join(experiment_dir, "Fold*"))]

    preproc_folder = os.path.join(cfg["data_folder"], cfg["preproc_name"])
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    split_info = json.load(open(split_file, "r"))
    test_split = split_info["test"]#[0:256]

    net_config = cfg
    loss_config = cfg["loss"]
    optim_config = cfg["optimizer"]

    test_ds = LVNCDataset(preproc_folder, test_split)
    test_cfg_dl = cfg["data_loader"].get("test", cfg["data_loader"]["val"]).copy()

    # --- Set batch size based on argument --------------------------
    if batch_size is not None:
        logger.info(f"Overriding batch size to {batch_size}")
        test_cfg_dl["batch_size"] = batch_size
    elif selected_device.type == "cpu":
        logger.info("Forcing batch_size=1 on CPU for detailed per-slice profiling")
        test_cfg_dl["batch_size"] = 1

    test_dl = DataLoader(test_ds, **test_cfg_dl)
    logger.info(f"Test dataloader batch size: {test_dl.batch_size}")

    # --- Initialize Model ------------------------------------------
    logger.info("Initializing neural network...")
    network = TransUNet2DEnsemble(
        checkpoint_paths,
        config=net_config,
        loss_config=loss_config,
        optim_config=optim_config,
        device=selected_device
    )
    
    
     # Apply channels_last memory format (if applicable)
    if selected_device.type in ("cuda", "cpu"):
        try:
            network = network.to(memory_format=torch.channels_last)
            logger.info("Applied channels_last memory format.")
        except Exception as e:
            logger.warning(f"Could not apply channels_last: {e}")

    if hasattr(torch, 'compile'):
        try:
            # Only compile if g++ is available (Linux)
            import shutil
            if shutil.which("g++") is not None:
                network = torch.compile(network)
                logger.info("Model compiled successfully with torch.compile().")
            else:
                logger.warning("Skipping torch.compile(): No C++ compiler found.")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}")

    test_segment_results_cb = StoreTestSegmentationResults(
        output_folder=os.path.join(results_dir, "output"),
        classes=net_config["unet"]["classes"],
        test_slice_list=test_split,
        output_size=(800, 800),
        raw_folder=os.path.join(cfg["data_folder"], "new_raw_dataset"),
        save_images = False,
        preproc_folder=preproc_folder,
        **cfg["callbacks"]["test_segmentation_results"]["params"],
        compute_volume_diff=False
    )

    # Clean up conflicting keys from config
    trainer_config = cfg["trainer"].copy()
    trainer_config.pop("accelerator", None)
    trainer_config.pop("devices", None)
    trainer_config.pop("precision", None)

    trainer = Trainer(
        callbacks=[test_segment_results_cb],
        logger=None,
        default_root_dir=results_dir,
        accelerator="gpu" if selected_device.type == "cuda" else "cpu",
        devices=1,
        precision=precision,
        **trainer_config
    )

    # --- Start Profiling ------------------------------------------
    logger.info("Starting profiling...")

    process = psutil.Process(os.getpid())
    max_ram_usage = 0
    max_cpu_usage = 0

    # GPU profiling (only if using GPU)
    if selected_device.type == "cuda":
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            max_gpu_mem = 0
            gpu_available = True
        except:
            logger.warning("Failed to initialize pynvml for GPU monitoring.")
            gpu_available = False
    else:
        gpu_available = False

    start_time = time.time()
    trainer.test(network, dataloaders=test_dl)
    end_time = time.time()

    tiempo_total = end_time - start_time
    num_cortes = len(test_dl.dataset)
    avg_time_per_slice = tiempo_total / num_cortes if num_cortes > 0 else float('nan')
    avg_time_per_batch = tiempo_total / test_cfg_dl["batch_size"]

    # --- Sample CPU and RAM ---------------------------------------
    for _ in range(10):
        max_ram_usage = max(max_ram_usage, process.memory_info().rss / (1024 * 1024))  # MB
        max_cpu_usage = max(max_cpu_usage, process.cpu_percent(interval=0.1))

    if gpu_available:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        max_gpu_mem = mem_info.used / (1024 * 1024)  # MB
        pynvml.nvmlShutdown()
    else:
        max_gpu_mem = "N/A"

    # --- Save Profiling Data --------------------------------------
    profiling_file = os.path.join(results_dir, f"profiling_results_{precision}_{batch_size}_{device}.csv")
    with open(profiling_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "total_time_sec", "avg_time_per_slice_sec", "avg_time_per_batch",
            "batch_size", "precision", "max_cpu_percent", "max_ram_mb", "max_gpu_mem_mb"
        ])
        writer.writerow([
            f"{tiempo_total:.4f}", f"{avg_time_per_slice:.4f}", f"{avg_time_per_batch:.4f}",
            f"{test_dl.batch_size}", precision, f"{max_cpu_usage:.2f}",
            f"{max_ram_usage:.2f}", f"{max_gpu_mem}"
        ])

    logger.success(f"Profiling completed. Results saved to {profiling_file}")
    
    # --- Compute Dice Coefficients -----------------------------------
    logger.info("Calculating Dice coefficients per slice on test set...")
    
    network = network.to(device=selected_device)
    network.eval()
    
    dice_df = pd.DataFrame(columns=['paciente', 'num_slice'] + net_config['unet']['classes'])
    
    with torch.no_grad():
        for batch in test_dl:
            pacientes = [str(p) for p in batch['patient']]
            slices = [int(s.item()) if isinstance(s, torch.Tensor) else s for s in batch['num_slice']]
            imgs = batch['image'].to(device=selected_device)
            masks = batch['mask'].to(device=selected_device)
    
            preds = network(imgs)
    
            pred_onehot, target_onehot = logits_to_onehot(preds, masks, len(net_config['unet']['classes']))
            dice_scores = compute_dice(pred_onehot, target_onehot, num_classes=len(net_config['unet']['classes']))
    
            df_batch = pd.DataFrame(dice_scores.cpu().numpy(), columns=net_config['unet']['classes'])
            df_batch.insert(0, 'num_slice', slices)
            df_batch.insert(0, 'paciente', pacientes)
    
            dice_df = pd.concat([dice_df, df_batch], ignore_index=True)
    
     # Ordenar y guardar coeficientes Dice por corte
    dice_df.sort_values(by=['paciente', 'num_slice'], inplace=True)
    dice_csv_path = os.path.join(results_dir, "dice_coefficients_test.csv")
    dice_df.to_csv(dice_csv_path, index=False)
    logger.success(f"Dice coefficients per slice saved to {dice_csv_path}")

    # --- Anadir columna 'set' desde df_info --------------------------
    #raw_data_folder = os.path.join(cfg["data_folder"], "new_raw_dataset_cleaned_via")
    raw_data_folder = os.path.join(cfg["data_folder"], "new_raw_dataset")
    df_info = pd.read_pickle(os.path.join(raw_data_folder, "df_info.pick"))

    # Unimos por paciente y slice
    dice_df = pd.merge(
        dice_df,
        df_info[['patient', 'slice', 'set']],
        left_on=['paciente', 'num_slice'],
        right_on=['patient', 'slice'],
        how='left'
    ).drop(columns=['patient', 'slice'])

    # --- Estadisticas globales por clase (todas las poblaciones) -----
    dice_stats_total = dice_df[net_config['unet']['classes']].agg(['mean', 'std']).T.round(4)
    dice_stats_total = dice_stats_total.rename(columns={"mean": "Dice_mean", "std": "Dice_std"})
    dice_stats_total.index.name = 'Clase'
    stats_total_path = os.path.join(results_dir, "dice_summary_total.csv")
    dice_stats_total.to_csv(stats_total_path)
    logger.success(f"Dice mean/std for full test set saved to {stats_total_path}")

    # --- Estadisticas por conjunto de poblacion (set) ----------------
    stats_by_set = []
    for set_value, df_group in dice_df.groupby("set"):
        stats = df_group[net_config['unet']['classes']].agg(['mean', 'std']).T.round(4)
        stats['Set'] = set_value
        stats.reset_index(inplace=True)
        stats.rename(columns={"mean": "Dice_mean", "std": "Dice_std", "index": "Clase"}, inplace=True)
        stats_by_set.append(stats)

    stats_by_set_df = pd.concat(stats_by_set, ignore_index=True)
    stats_by_set_path = os.path.join(results_dir, "dice_summary_by_set.csv")
    stats_by_set_df.to_csv(stats_by_set_path, index=False)
    logger.success(f"Dice mean/std per class by population set saved to {stats_by_set_path}")



if __name__ == "__main__":
    app()