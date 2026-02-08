import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

from PIL import Image
from skimage import color
from skimage.transform import resize
from torchvision import transforms
import skimage.exposure as exposure
import cv2 as cv
from ultralytics import YOLO
from loguru import logger


# https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d
# https://docs.opencv.org/3.4/d5/d98/tutorial_mat_operations.html
def s16_to_u8(data):
    if data.dtype==np.uint8:
        return data
    alpha = 255 / (np.max(data) - np.min(data))
    beta = -np.min(data) * alpha
    return abs(data * alpha + beta).astype(np.uint8)

def dicom_to_image(dicom_array, target_size=None):
    image = s16_to_u8(dicom_array)
    if len(image.shape) == 3:
        image = (255*color.rgb2gray(image)).astype(np.uint8)
    if target_size is not None and image.shape!=target_size:
        image = resize(image, target_size, order=3, anti_aliasing=False, preserve_range=True)
    return image

def generate_split_dict(df_train: pd.DataFrame, df_test: pd.DataFrame, column_stratify, n_folds = 5, shuffle=True, seed=1234, query="tuple()", query_before_cv = False):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    cross_val = []
    if query_before_cv:
        df_train = df_train.query(query)
    for train_index, val_index in skf.split(df_train, df_train[column_stratify]):
        df_t = df_train.iloc[train_index]
        df_v = df_train.iloc[val_index]
        if not query_before_cv:
            df_t = df_t.query(query)
            df_v = df_v.query(query)
        cross_val.append({
            "train": [(row["patient"], row["slice"]) for _, row in df_t.iterrows()],
            "val": [(row["patient"], row["slice"]) for _, row in df_v.iterrows()]
        })
    return {
        'test': [(row["patient"], row["slice"]) for _, row in df_test.iterrows()],
        "cross_validation": cross_val
    }

def combine_images(back, front, alpha=50):
    back = back.convert('RGBA')
    front = front.convert('RGBA')
    front.putalpha(alpha) # Half alpha
    data = np.array(front)

    r1, g1, b1,a1 = 0, 0, 0,50 # Original value
    r2, g2, b2,a2 = 0, 0, 0,0 # Value that we want to replace it with

    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    mask = (red == r1) & (green == g1) & (blue == b1) & (alpha==a1)
    data[:,:,:4][mask] = [r2, g2, b2,a2]

    front = Image.fromarray(data)

    composite = Image.alpha_composite(back, front)
    
    return back.convert('RGB'), composite.convert('RGB')
    
    
    
def pipeline_preprocessing_mri(roi, target_size = (800, 800)):
    '''
    Dada una region de interes, empleamos tecnicas de preprocesamiento de imagenes
    para eliminar ruido y mejorar el contraste.
    '''
    
    # 1) Normalizar el rango de intensidad [0..255]
    #    Esto asegura que la ROI use todo el rango dinámico disponible.
    enhanced_roi = exposure.rescale_intensity(
        roi, in_range='image', out_range=(0, 255)
    ).astype(np.uint8)
    
    # Extraemos dimensiones y centro (por si fueran útiles después)
    h, w = enhanced_roi.shape
    center_roi = (w // 2, h // 2)
    
    # 2) Filtrado Bilateral:
    #    Ajustamos los parámetros según el tamaño de la ROI para
    #    preservar bordes anatómicos y reducir ruido.
    d = 7
    sigmaColor = 75
    sigmaSpace = 75
    tileGridSize = (2, 2)
        
    roi_bilateral_filtered = cv.bilateralFilter(
        enhanced_roi, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
    )


    # 3) CLAHE (ecualización adaptativa):
    #    - clipLimit controla la sobre-ecualización.
    #    - tileGridSize define el número de sub-bloques
    #      para distribuir mejor el histograma local.
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=tileGridSize)
    roi_clahe = clahe.apply(roi_bilateral_filtered)
    
    return roi_clahe
