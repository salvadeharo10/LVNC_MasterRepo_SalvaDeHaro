import os
from typing import List
from loguru import logger
import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import glob
import pickle

from config import load_config_files
from data.dataset import LVNCDataset
from modules.transUnet import TransUNet, TransUNet2DEnsemble

from data.maps_utils import logits_to_onehot, image_from_onehot

from PIL import Image
from palettable.cartocolors import qualitative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------------
original_cfg_dict, cfg = load_config_files(
    ['./config_files/config_2025.yaml', './config_files/config_2025.yaml']
)
preproc_folder = os.path.join(cfg["data_folder"], cfg["preproc_name"])
split_json_path = os.path.join(
    cfg["data_folder"],
    'split_5_cv_HCM_X_Hebron_Titina_groupedby_patient.json'
)

with open(split_json_path, "r") as f:
    split_info = json.load(f)


# Read dataframe with information of slices
raw_folder = os.path.join(cfg["data_folder"], "new_raw_dataset")
df_info = pd.read_pickle(os.path.join(raw_folder, "df_info.pick"))

# Controla si se guardan o no las segmentaciones coloreadas
SAVE_COLORED_SEGMENTATIONS = True  # <--- cambia a True si quieres activarlo
OVERLAY_ALPHA = 50                  # transparencia de las máscaras (0-255)

# Paleta de colores para las clases (compatible con image_from_onehot)
# (se pasa después a la función aquiles)
# Se define más abajo cuando ya sabemos 'classes'.

# -------------------------------------------------------------------------
# Definición de splits
# -------------------------------------------------------------------------
cross_val_split = split_info['cross_validation']
test_split = split_info['test']

train_fold0 = cross_val_split[0]['train']
val_fold0 = cross_val_split[0]['val']

# Dataset total = train + val + test
all_slices = train_fold0 + val_fold0 + test_split
#all_slices = test_split

dataset_lvnc = LVNCDataset(preproc_folder, all_slices)

cfg_dl = cfg["data_loader"]["train"]
dataloader_lvnc = DataLoader(dataset_lvnc, **cfg_dl)

config_net = cfg
classes = config_net['unet']['classes']

# Definir paleta de colores
palette = None#qualitative.get_map(f"Bold_{len(classes)}").colors

patches_size = config_net['transformer']["patch_size"]
config_net['transformer']["patches"]["grid"] = (
    int(800 / patches_size),
    int(800 / patches_size),
)
loss_config = cfg["loss"]
optim_config = cfg["optimizer"]

checkpoint_paths = []
for fold_dir in glob.glob(os.path.join('./logs/vitu_normal/', "Fold*")):
    checkpoint_paths.append(glob.glob(os.path.join(fold_dir, "epoch*.ckpt"))[0])

#checkpoint_paths = sorted(
#    glob.glob('./logs/vitunet_lvnc_detector_via/fold*.ckpt')
#)


network = TransUNet2DEnsemble(
    checkpoint_paths, config=config_net,
    loss_config=loss_config, optim_config=optim_config, device = device
)

# -------------------------------------------------------------------------
# Función para calcular Dice
# -------------------------------------------------------------------------
def compute_dice(
    output: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = None,
    compute_classes: List[int] = None,
    smooth=1e-8
):
    """
    output and target must be onehot representations
        NxCxHxW
    """
    if compute_classes is None:
        compute_classes = list(range(num_classes))

    output = output[:, compute_classes, :, :]
    target = target[:, compute_classes, :, :]

    intersection = torch.mul(output, target)
    dice = (2 * torch.sum(intersection, dim=(2, 3)) + smooth) / (
        torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) + smooth
    )

    return dice


# -------------------------------------------------------------------------
# Evaluación + (opcional) guardado de overlays coloreados
# -------------------------------------------------------------------------
def aquiles(
    dataloader,
    model,
    classes,
    save_segmentations: bool = False,
    seg_output_dir: str = None,
    preproc_folder: str = None,
    palette=None,
    overlay_alpha: int = 50,
):
    """
    Si save_segmentations=True, guarda en seg_output_dir imágenes PNG
    con la segmentación coloreada superpuesta sobre la imagen original.
    """
    model = model.to(device=device)
    model.eval()

    num_classes = len(classes)
    df_global = pd.DataFrame(columns=['paciente', 'num_slice'] + classes)

    if save_segmentations:
        if seg_output_dir is None:
            raise ValueError("seg_output_dir no puede ser None si save_segmentations=True")
        if preproc_folder is None:
            raise ValueError("preproc_folder no puede ser None si save_segmentations=True")
        Path(seg_output_dir).mkdir(parents=True, exist_ok=True)
        if palette is None:
            #palette = qualitative.get_map(f"Bold_{num_classes}").colors
            palette = [
                [0,   0,   0],   # clase 1 background
                [0,   0,   255],   # clase 2 inner cavity
                [0,     255, 0],   # clase 3 external layer
                [255, 0,   0],   # clase 4 trabeculated zone
            ]

    with torch.no_grad():
        for batch in dataloader:
            patients = batch['patient']
            num_slices = batch['num_slice']

            imgs = batch['image'].clone().detach().to(device=device)
            targets = batch['mask'].clone().detach().to(device=device)

            preds = model(imgs)

            pred_onehot, target_onehot = logits_to_onehot(preds, targets, num_classes)
            dice = compute_dice(pred_onehot, target_onehot, num_classes=num_classes)

            df_actual = pd.DataFrame(dice.cpu().numpy(), columns=classes)
            df_actual.insert(0, 'paciente', patients)
            df_actual.insert(1, 'num_slice', num_slices)
            df_global = pd.concat([df_global, df_actual], ignore_index=True)

            # -------------------------------------------------------------
            # Guardado de segmentaciones coloreadas (opcional)
            # -------------------------------------------------------------
            if save_segmentations:
                # pred_onehot: (N, C, H, W)
                pred_onehot_np = pred_onehot.cpu().numpy()

                for i in range(pred_onehot_np.shape[0]):
                    pat = patients[i]
                    slc = num_slices[i]

                    # Asegurarnos de tener tipos nativos para construir el nombre
                    if isinstance(pat, torch.Tensor):
                        pat = pat.item()
                    if isinstance(slc, torch.Tensor):
                        slc = int(slc.item())
                    else:
                        slc = int(slc)

                    # 1. Máscara coloreada desde el one-hot
                    mask_i = pred_onehot_np[i]  # (C, H, W)
                    seg_rgb = image_from_onehot(mask_i, palette, skip_bg=True)

                    # 2. Imagen de fondo (preprocesada original)
                    img_path = os.path.join(preproc_folder, "images", f"{pat}_{slc}.pick")
                    if not os.path.isfile(img_path):
                        logger.warning(f"No se encuentra la imagen preprocesada: {img_path}")
                        continue
                    
                    img = pickle.load(open(img_path, "rb"))
                    img = np.array(img)
                    
                    # --- NORMALIZACIÓN PARA VISUALIZACIÓN ---
                    # Si la imagen no está en uint8, la reescalamos a [0, 255]
                    if img.dtype != np.uint8:
                        vmin, vmax = np.min(img), np.max(img)
                        if vmax > vmin:
                            # Si parece estar en [0, 1], esto sigue funcionando igual
                            img_norm = (img - vmin) / (vmax - vmin + 1e-8)
                        else:
                            # Imagen constante, la ponemos negra
                            img_norm = np.zeros_like(img, dtype=np.float32)
                        img_vis = (img_norm * 255).clip(0, 255).astype(np.uint8)
                    else:
                        img_vis = img
                    
                    back = Image.fromarray(img_vis).convert("RGBA")


                    # 3. Capa frontal con NEAREST para no introducir grises
                    front = Image.fromarray(seg_rgb).resize(back.size, resample=Image.NEAREST)
                    front = front.convert("RGBA")

                    data = np.array(front)

                    # 4. Construir canal alfa:
                    #    - fondo negro (0,0,0) -> alfa 0
                    #    - clases coloreadas    -> alfa overlay_alpha
                    rgb = data[:, :, :3]
                    is_bg = (rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0)
                    alpha = np.where(is_bg, 0, overlay_alpha).astype(np.uint8)
                    data[:, :, 3] = alpha

                    front = Image.fromarray(data, mode="RGBA")

                    composite = Image.alpha_composite(back, front)
                    out_name = f"{pat}_{slc:02d}_overlay.png"
                    composite.convert("RGB").save(os.path.join(seg_output_dir, out_name))

        df_sorted = df_global.sort_values(
            by=['paciente', 'num_slice'],
            ascending=[True, True]
        ).reset_index(drop=True)
        return df_sorted


# -------------------------------------------------------------------------
# Helpers para clasificar pacientes
# -------------------------------------------------------------------------
def get_hard_patients(df, umbral):
    return df[df['t'] < umbral]


def get_candy_patients(df, umbral):
    return df[df['t'] >= umbral]


# -------------------------------------------------------------------------
# Ejecución principal
# -------------------------------------------------------------------------
umbral = 0.5
csv_dir = './diceCoef_vitu_normal'
Path(csv_dir).mkdir(parents=True, exist_ok=True)

# Carpeta donde se guardarán las segmentaciones coloreadas
seg_output_dir = os.path.join(csv_dir, 'segmentaciones_coloreadas')

dice_coefficients_df = aquiles(
    dataloader_lvnc,
    network,
    classes,
    save_segmentations=SAVE_COLORED_SEGMENTATIONS,
    seg_output_dir=seg_output_dir,
    preproc_folder=preproc_folder,
    palette=palette,
    overlay_alpha=OVERLAY_ALPHA,
)

dice_coefficients_df = pd.merge(
    dice_coefficients_df,
    df_info[['patient', 'slice', 'set']],
    left_on=['paciente', 'num_slice'],
    right_on=['patient', 'slice'],
    how='left'
).drop(columns=['patient', 'slice'])

# Guardar el DataFrame en un archivo CSV
dice_coefficients_df.to_csv(
    os.path.join(csv_dir, 'dice_coefficients_slices_patients.csv'),
    index=False
)

# Si quieres activar los CSV de hard/candy:
# df_hard_patients = get_hard_patients(dice_coefficients_df, umbral)
# df_hard_patients.to_csv(os.path.join(csv_dir, 'hard_slices_patients.csv'), index=False)
# df_candy_patients = get_candy_patients(dice_coefficients_df, umbral)
# df_candy_patients.to_csv(os.path.join(csv_dir, 'candy_slices_patients.csv'), index=False)
