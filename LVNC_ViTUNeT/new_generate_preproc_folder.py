import os
from PIL import Image
import typer
from shutil import copy, SameFileError
from pathlib import Path
import pickle
from typing import List, Tuple
from loguru import logger

import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from pydicom import dcmread
import cv2 as cv

#from data.preprocessing_with_yolo import Preprocessor2DYOLO
from data.preprocessing import Preprocessor2D
from data.maps_utils import compute_pta, image_from_onehot
from data.utils import combine_images

import torch

app = typer.Typer()

@app.command()
def run(
    #Esta opción es obligatoria y representa la carpeta que contiene el conjunto de datos a procesar. Debe existir y ser legible.
    data_folder: Path = typer.Option(...,
                                    help="Folder containing the dataset",
                                    exists=True,
                                    readable=True),
    
    # Esta opción es obligatoria y especifica la carpeta de destino que contendrá el conjunto de datos preprocesado.                                
    preproc_name: str = typer.Option(..., help="Destination folder which will contain the preprocessed dataset"),
    
    # Esta opción es obligatoria y especifica el tamaño de salida deseado para las imágenes y las segmentaciones de referencia. Debe proporcionarse como una tupla de dos enteros.
    target_size: Tuple[int, int] = typer.Option(..., help="Output size for images and ground truth segmentations"),
    
    proc_mri_image: bool = typer.Option(False, help="Process MRI images with some filters of artificial vision"),
   
    #Esta opción es opcional y especifica el método utilizado para normalizar las imágenes.
    normalization_method: str = typer.Option("zscore", help="Method used to normalize the image"),
   
    # Esta opción es opcional y es un indicador booleano que determina si la clase de cavidad interna (asumida como 2) debe ser eliminada.
    rm_ic: bool = typer.Option(False, help="Whether or not internal cavity class (assumed 2) should be removed"),
    
    # Esta opción es opcional y es un indicador booleano que determina si las segmentaciones deben convertirse en máscaras multietiqueta.
    multi_label: bool = typer.Option(False, help="Whether or not segmentations must be converted to multi-label masks"),
   
    # Esta opción es opcional y es un indicador booleano que determina si se debe crear una carpeta llamada vis_gt que contendrá las segmentaciones de referencia dibujadas sobre las imágenes de entrada.
    produce_visual_gt: bool = typer.Option(False, help="Whether or not a folder `vis_gt` containing GT drawn over the input images must be created"),
   
    # Esta opción es opcional y especifica la extensión del formato de las imágenes de segmentación de referencia generadas si produce_visual_gt es True.
    vis_gt_format: str = typer.Option("jpg", help="Extension of visual gt if generated"),
    
    #: Esta opción es opcional y es un indicador booleano que determina si se debe utilizar la interpolación de vecino más cercano para cambiar el tamaño de la máscara.
    nearest_neigh_mask: bool = typer.Option(False, help="Wether or not nearest neighbor interpolation should be used to resize the mask"),
    
    # Esta opción es opcional y es un indicador booleano que determina si se debe utilizar el redimensionamiento combinado para cambiar el tamaño de la máscara. Si se establece en True, nearest_neigh_mask se ignora.
    combined_resize: bool = typer.Option(False, help="Wether or not combined resize should be used to resize the mask. If true, nearest_neigh_mask is ignored"),
):
    raw_folder = os.path.join(data_folder, "new_raw_dataset_cleaned")
    output_folder = os.path.join(data_folder, preproc_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    logger.add(os.path.join(output_folder, "_log.txt"))

    logger.info(f"Raw data folder: {raw_folder}")
    logger.info(f"Preprocessed folder: {output_folder}")
    logger.info(f"Target size: {target_size}")
    logger.info(f"Preprocess MRI images: {proc_mri_image}")
    logger.info(f"Normalization method: {normalization_method}")
    logger.info(f"Remove internal cavity: {rm_ic}")
    logger.info(f"Multi-label: {multi_label}")
    logger.info(f"Produce visual GT: {produce_visual_gt}")
    logger.info(f"Nearest neighbor for mask resizing: {nearest_neigh_mask}{' (ignored)' if combined_resize else ''}")
    logger.info(f"Combined resize: {combined_resize}")
        
    # Create necessary folders
    images_folder = os.path.join(output_folder, "images")
    gt_folder = os.path.join(output_folder, "gt")
    Path(images_folder).mkdir(parents=True, exist_ok=True)
    Path(gt_folder).mkdir(parents=True, exist_ok=True)
    if produce_visual_gt:
        vis_gt_folder = os.path.join(output_folder, "vis_gt")
        Path(vis_gt_folder).mkdir(parents=True, exist_ok=True)
    
    
    # Instantiate preprocessor
    preprocessor = Preprocessor2D(target_size=target_size, normalization_method=normalization_method, rm_ic=rm_ic, multi_label=multi_label, compute_pta=True, compute_area_ic = True, nearest_neigh_mask=nearest_neigh_mask, combined_resize=combined_resize, proc_mri_image = proc_mri_image)
    #preprocessor = Preprocessor2DYOLO(target_size=target_size, output_folder = output_folder, normalization_method=normalization_method, rm_ic=rm_ic, multi_label=multi_label, compute_pta=True, nearest_neigh_mask=nearest_neigh_mask, combined_resize=combined_resize, proc_mri_image = proc_mri_image)

    # Read dataframe with information of slices
    df_info = pd.read_pickle(os.path.join(raw_folder, "df_info.pick")).set_index(["patient", "slice"])
    
    def preproc_slice(patient, slice):
        #dicom_path = os.path.join(raw_folder, f"dicom/{patient}_{slice}.dcm")
        jpg_path = os.path.join(raw_folder, f"jpg/{patient}_{slice}.jpg")

        #if os.path.exists(dicom_path):
            #dcm = dcmread(dicom_path)
        if os.path.exists(jpg_path):
            # Use PIL to read the JPEG image
            #img = np.array(Image.open(jpg_path))
            img = cv.cvtColor(cv.imread(jpg_path), cv.COLOR_BGR2RGB)
            # Create a dummy DICOM object for consistency in the rest of the code
            dcm = DummyDICOMObject(pixel_array=img)
        else:
            raise FileNotFoundError(f"Neither DICOM nor JPEG file found for {patient}_{slice}")

        img = dcm.pixel_array
        gt = pickle.load(open(os.path.join(raw_folder, f"segmentation/{patient}_{slice}.pick"), "rb"))
       
            
        img_p, gt_p, pta, ape, at, aic = preprocessor(img, gt)

        
        pickle.dump(img_p, open(os.path.join(images_folder, f"{patient}_{slice}.pick"), "wb"))
        pickle.dump(gt_p, open(os.path.join(gt_folder, f"{patient}_{slice}.pick"), "wb"))

        if produce_visual_gt:
            colors = [[0, 0, 0], [17, 165, 121], [67, 191, 222], [242, 183, 1]]
            gt_onehot = (np.arange(gt_p.max() + 1) == gt_p[..., None]).astype(int).transpose((2, 0, 1))
            front = image_from_onehot(gt_onehot, colors)
            back, composite = combine_images(Image.fromarray(img_p), Image.fromarray(front), alpha=25)
            composite.save(os.path.join(vis_gt_folder, f"{patient}_{slice}.{vis_gt_format}"))

        return pta, ape, at, aic  # rh and rv do not affect PTA value
    

    # DummyDICOMObject class to create a fake DICOM object for consistency
    class DummyDICOMObject:
        def __init__(self, pixel_array):
            self.pixel_array = pixel_array


    # Process the entire dataset
    with Parallel(n_jobs=max(1, 1)) as parallel:#with Parallel(n_jobs=max(1, multiprocessing.cpu_count()-1)) as parallel:
        #computed_ptas, computed_apes, computed_ats, _ = parallel(delayed(preproc_slice)(p, s) for (p,s), _ in tqdm(df_info.iterrows()))
        resultados = parallel(delayed(preproc_slice)(p, s) for (p,s), _ in tqdm(df_info.iterrows()))
    
    # Extraer los valores de pta y ape de las ternas en resultados
    computed_ptas = [pta for pta, _, _, _ in resultados]
    computed_apes = [ape for _, ape, _, _ in resultados]
    computed_ats = [at for _, _, at, _ in resultados]
    computed_aics = [aic for _, _, _, aic in resultados]


    df_results = pd.DataFrame(index=df_info.index)
    df_results["reference_pta"] = computed_ptas
    df_results['reference_ape'] = computed_apes
    df_results['reference_at'] = computed_ats
    df_results['reference_aic'] = computed_aics
    df_results.to_pickle(os.path.join(output_folder, "preproc_results.pick"))

if __name__=="__main__":
    #torch.backends.cudnn.benchmark = False
    app()
