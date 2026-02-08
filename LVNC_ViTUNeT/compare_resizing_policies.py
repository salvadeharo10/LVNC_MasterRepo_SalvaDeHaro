#%%
import os
import pandas as pd
import pickle
import numpy as np
from pydicom import dcmread
from PIL import Image
from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

import torch
import torchvision
from albumentations.augmentations.geometric.resize import Resize

import cv2
from skimage import color
from skimage.transform import resize

from data.maps_utils import compute_pta, image_from_onehot, logits_to_onehot, resize_class_map
from data.utils import combine_images
from metrics.dice import compute_dice

#%%
RAW_DATA_FOLDER = "../LVNC_dataset/raw_dataset"
DICOM_FOLDER = os.path.join(RAW_DATA_FOLDER, "dicom")
SEGMENTATION_FOLDER = os.path.join(RAW_DATA_FOLDER, "segmentation")
OUTPUT_FOLDER = "/src/data/test_resizing_policies"

#%% Load the dataset
df_info = pd.read_pickle(os.path.join(RAW_DATA_FOLDER, "df_info.pick"))

#%%
# https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d
# https://docs.opencv.org/3.4/d5/d98/tutorial_mat_operations.html
def s16_to_u8(data):
    if data.dtype==np.uint8:
        return data
    alpha = 255 / (np.max(data) - np.min(data))
    beta = -np.min(data) * alpha
    return abs(data * alpha + beta).astype(np.uint8)


#%%
def resize_map_NN(image, gt, target_size=(256, 256), anti_aliasing=False, clip=True, **kwargs):
    if target_size is None:
        target_size = gt.shape
    image = resize(image, target_size, order=3, anti_aliasing=False, preserve_range=True)
    gt = resize(gt.astype(float), target_size, order=0, mode="edge", clip=clip, anti_aliasing=anti_aliasing, **kwargs)
    return image, gt

def resize_with_albumentations(image, gt, target_size=(256, 256), **kwargs):
    rt = Resize(target_size[0], target_size[1], **kwargs)
    trans = rt(image=image, mask=gt)
    image = trans["image"]
    gt = trans["mask"]
    return image, gt

def resize_masks(image, gt, order=3, target_size=(256,256), **kwargs):
    gt_p = resize_class_map(gt, order=order, target_size=target_size, **kwargs)
    return resize_image(image, gt_p)

def combined_resize(image, gt, target_size=(256,256), **kwargs):
    image, gt_p1 = resize_masks(image, gt, target_size=target_size, **kwargs)
    _, gt_p2 = resize_map_NN(image, gt, target_size, anti_aliasing=False)
    mask=np.logical_and(gt_p1==0, np.logical_or(gt_p1==2, gt_p2==3))
    gt_p1[mask] = gt_p2[mask] # Asignamos para evitar eliminar los huecos negros que puedan aparecer
    return image, gt_p1

def resize_masks_NN(image, gt, target_size=(256,256), **kwargs):
    gt_p = resize_class_map(gt, order=0, target_size=target_size, **kwargs)
    return resize_image(image, gt_p)

def resize_image(image, gt, target_size=None):
    if target_size is None:
        target_size = gt.shape
    image = resize(image, target_size, order=3, anti_aliasing=False, preserve_range=True)

    return image, gt


#%%
def process_slice(patient, slc, transform, suffix, **kwargs):
    dcm = dcmread(os.path.join(DICOM_FOLDER, f"{patient}_{slc}.dcm"))
    gt = pickle.load(open(os.path.join(SEGMENTATION_FOLDER, f"{patient}_{slc}.pick"), "rb"))
    image = dcm.pixel_array
    image = s16_to_u8(image)
    if len(image.shape) == 3:
        image = (255*color.rgb2gray(image)).astype(np.uint8)

    image_p, gt_p = transform(image, gt, **kwargs)

    colors = [[0, 0, 0], [17, 165, 121], [67, 191, 222], [242, 183, 1]]
    gt_onehot=(np.arange(gt_p.max()+1) == gt_p[...,None]).astype(int).transpose((2,0,1))
    front = image_from_onehot(gt_onehot, colors)
    back, composite= combine_images(Image.fromarray(image_p), Image.fromarray(front), alpha=50)
    composite.save(os.path.join(OUTPUT_FOLDER, f"{patient}_{slc}_{suffix}.png"))
    composite.resize((800,800)).save(os.path.join(OUTPUT_FOLDER, f"{patient}_{slc}_{suffix}_800.png"))
    pickle.dump(gt_p, open(os.path.join(OUTPUT_FOLDER, f"{patient}_{slc}_{suffix}.pick"), "wb"))


#%%
# Antialias claramente no funciona bien en este caso así que lo comento para evitar perder tiempo de cómputo

def try_all(row):
    if False:
        process_slice(row["patient"], row["slice"], resize_image, "resize_img")
        process_slice(row["patient"], row["slice"], resize_map_NN, "256_NN")
        process_slice(row["patient"], row["slice"], resize_map_NN, "256_NN_noclip", clip=False)
        process_slice(row["patient"], row["slice"], resize_with_albumentations, "256_albumentations_NN", interpolation=cv2.INTER_NEAREST)
        process_slice(row["patient"], row["slice"], resize_map_NN, "512_NN", target_size=(512, 512))
        process_slice(row["patient"], row["slice"], resize_map_NN, "512_NN_noclip", clip=False, target_size=(512, 512))
        process_slice(row["patient"], row["slice"], resize_with_albumentations, "512_albumentations_NN", interpolation=cv2.INTER_NEAREST, target_size=(512, 512))

        process_slice(row["patient"], row["slice"], resize_masks_NN, "512_masks_NN_no_antialias", target_size=(512, 512), anti_aliasing=False)
        process_slice(row["patient"], row["slice"], resize_masks, "512_masks_3_no_antialias", order=3, target_size=(512, 512), anti_aliasing=False)
        process_slice(row["patient"], row["slice"], resize_masks_NN, "256_masks_NN_no_antialias", target_size=(256, 256), anti_aliasing=False)
        process_slice(row["patient"], row["slice"], resize_masks, "256_masks_3_no_antialias", order=3, target_size=(256, 256), anti_aliasing=False)
        process_slice(row["patient"], row["slice"], combined_resize, "256_combined_no_antialias", order=3, target_size=(256,256), anti_aliasing=False)
        process_slice(row["patient"], row["slice"], combined_resize, "512_combined_no_antialias", order=3, target_size=(512,512), anti_aliasing=False)

with Parallel(n_jobs=max(1, multiprocessing.cpu_count()-1)) as parallel:
    parallel(delayed(try_all)(row) for _, row in tqdm(df_info.iterrows()))

# %% Compute dices
def get_dice(patient, slc, suffix):
    gt = pickle.load(open(os.path.join(SEGMENTATION_FOLDER, f"{patient}_{slc}.pick"), "rb"))
    gt_p = pickle.load(open(os.path.join(OUTPUT_FOLDER, f"{patient}_{slc}_{suffix}.pick"), "rb"))
    # Obtener el dice tras volver a subir a los 800x800
    gt_p_p = resize(gt_p.astype(float), (800,800), order=0, mode="edge")
    gt_p_onehot, gt_onehot = logits_to_onehot(torch.tensor(gt_p_p, dtype=torch.int64).unsqueeze(0), torch.tensor(gt, dtype=torch.int64).unsqueeze(0), num_classes=4)
    return compute_dice(gt_p_onehot, gt_onehot, num_classes=4)

suffix_list = [
    "256_NN",
    #"256_NN_noclip",
    #"256_albumentations_NN",
    #"256_masks_NN_no_antialias",
    #"256_masks_3_no_antialias",
    "512_NN",
    #"512_NN_noclip",
    #"512_albumentations_NN",
    "512_masks_NN_no_antialias",
    "512_masks_3_no_antialias",
    "256_combined_no_antialias",
    "512_combined_no_antialias",
]
dice_results = {s: [] for s in suffix_list}
def dice_all(row):
    dr = {s: 0 for s in suffix_list}
    for s in dr:
        dr[s]=get_dice(row["patient"], row["slice"], s)
    return dr

with Parallel(n_jobs=4) as parallel:
    drs = parallel(delayed(dice_all)(row) for _, row in tqdm(df_info.iterrows()))

for dr in drs:
    for s in dice_results:
        dice_results[s].append(dr[s])

# %%
for s in dice_results:
    print(s, torch.stack(dice_results[s]).squeeze().mean(0))
    print(s, torch.stack(dice_results[s]).squeeze().median(0))

#%%
print((torch.stack(dice_results["512_NN"])-torch.stack(dice_results["256_NN"])).squeeze().min(0))
print((torch.stack(dice_results["512_NN"])-torch.stack(dice_results["256_NN"])).squeeze().max(0))


# %%
print((torch.stack(dice_results["512_masks_3_no_antialias"])-torch.stack(dice_results["512_NN"])).squeeze().min(0))
print((torch.stack(dice_results["512_masks_3_no_antialias"])-torch.stack(dice_results["512_NN"])).squeeze().max(0))

# %% Estos dos parece que hacen lo mismo
print((torch.stack(dice_results["512_masks_NN_no_antialias"])-torch.stack(dice_results["512_NN"])).squeeze().min(0))
print((torch.stack(dice_results["512_masks_NN_no_antialias"])-torch.stack(dice_results["512_NN"])).squeeze().max(0))
# %%
print((torch.stack(dice_results["512_combined_no_antialias"])-torch.stack(dice_results["512_masks_3_no_antialias"])).squeeze().min(0))
print((torch.stack(dice_results["512_combined_no_antialias"])-torch.stack(dice_results["512_masks_3_no_antialias"])).squeeze().max(0))

# %%
print((torch.stack(dice_results["512_combined_no_antialias"])-torch.stack(dice_results["256_combined_no_antialias"])).squeeze().min(0))
print((torch.stack(dice_results["512_combined_no_antialias"])-torch.stack(dice_results["256_combined_no_antialias"])).squeeze().max(0))