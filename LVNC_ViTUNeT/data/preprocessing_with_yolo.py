import numpy as np
from skimage import color
from skimage.transform import resize

from .maps_utils import resize_class_map, compute_pta
from .utils import s16_to_u8, pipeline_preprocessing_mri

from ultralytics import YOLO
from yolo2.yoloDetect import process_image_and_detect_with_yolo
import cv2
from loguru import logger
import os

def remove_classes(segmentation, rm_classes):
    """
    It works with any shape of `segmentation`
    `rm_classes` must be sorted from the highest to the lowest index
    """
    for c in rm_classes:
        mask_rm = segmentation==c
        mask_change = segmentation>c 
        segmentation[mask_rm]=0
        segmentation[mask_change] = segmentation[mask_change] - 1
    
    return segmentation


class Preprocessor2DYOLO(object):
    def __init__(self, target_size, output_folder, normalization_method="zscore", rm_ic=False, multi_label = False, use_mask=False, compute_pta = False, epsilon=1e-8, nearest_neigh_mask=False, combined_resize=False, proc_mri_image = False):
        self.normalization_method = normalization_method
        self.rm_ic = rm_ic
        self.multi_label = multi_label
        self.use_mask = use_mask
        self.target_size = target_size
        self.compute_pta = compute_pta
        self.epsilon = epsilon
        self.nn_mask = nearest_neigh_mask
        self.combined_resize = combined_resize # If true, ignore nn_mask
        self.proc_mri_image = proc_mri_image
        self.yolo_model = YOLO('./yolo2/best_IC_genZ.pt')
        
        logger.add(os.path.join(output_folder, "_log2.txt"))


    def __call__(self, dicom_array, segmentation):
        #YOLO DETECTION
        image = s16_to_u8(dicom_array) #yolo model was trained with images in uint8 and RGB

        img_yolo, seg_yolo = process_image_and_detect_with_yolo(self.yolo_model, image, segmentation)

        if img_yolo is None: #si el modelo YOLO ha fallado...
            # Encontrar contorno cavidad interna en la máscara
            contours_ic, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Encontrar el contorno más grande (asumiendo que es la cavidad interna)
            largest_contour = max(contours_ic, key=cv2.contourArea)
            x1, y1, w, h = cv2.boundingRect(largest_contour) #(x,y) vertice superior izquierdo
            x2 = x1 + w
            y2 = y1 + h

            factor_escala_y = image.shape[0]/segmentation.shape[0]
            factor_escala_x = image.shape[1]/segmentation.shape[1]

            x11 = int(x1*factor_escala_x)
            y11 = int(y1*factor_escala_y)
            x22 = int(x2*factor_escala_x)
            y22 = int(y2*factor_escala_y)

            img_yolo = image[y11:y22, x11:x22]
            seg_yolo = segmentation[y1:y2, x1:x2]
            logger.info(f"Entro aqui y me cago en mis muertos {img_yolo.shape}")

        logger.info(f"HOla {img_yolo.shape}")
        
    
        if self.combined_resize:
            segmentation_r = resize_class_map(seg_yolo, order=3, target_size=self.target_size)
            segmentation_nn = resize(seg_yolo.astype(float), self.target_size, order=0, mode="edge", anti_aliasing=False)
            mask=np.logical_and(segmentation_r==0, np.logical_or(segmentation_nn==2, segmentation_nn==3))
            segmentation_r[mask] = segmentation_nn[mask] # Replace "black points" with the value of the NN resize if it contains interval cavity or trabeculae
            seg_yolo = segmentation_r
        else:
            seg_yolo = resize_class_map(seg_yolo, target_size=self.target_size, order = 0 if self.nn_mask else 3)


        if self.compute_pta:
            pta, ape, at = compute_pta(seg_yolo, rh=1, rv=1)
        else:
            pta = None
            ape = None
            at = None

        if self.multi_label:
            # One class containing the external wall and the trabeculae
            # Another class with the trabeculae
            binary_masks = np.zeros((2 if self.rm_ic else 3, *self.target_size), np.uint8)
            binary_masks[0, seg_yolo==1] = 1
            binary_masks[0, seg_yolo==3] = 1
            if self.rm_ic:
                binary_masks[1, seg_yolo==3] = 1
            else:
                binary_masks[1, seg_yolo==2] = 1
                binary_masks[2, seg_yolo==3] = 1

            seg_yolo = binary_masks
        else:
            if self.rm_ic:
                seg_yolo = remove_classes(seg_yolo, [2])

        
        if len(img_yolo.shape) == 3:
            img_yolo = (255*color.rgb2gray(img_yolo)).astype(np.uint8)
        if img_yolo.shape!=self.target_size:
            img_yolo = resize(img_yolo, self.target_size, order=3, anti_aliasing=False, preserve_range=True)
        if self.proc_mri_image: # Arreglo de la ROI obtenida con tecnicas de filtrado
            img_yolo = pipeline_preprocessing_mri(img_yolo, target_size = self.target_size)
        if self.normalization_method=="zscore":
            if self.use_mask:
                mask = seg_yolo > 0
            else:
                mask = np.ones(seg_yolo.shape, dtype=bool)
            img_yolo = (img_yolo - img_yolo.mean())/(img_yolo.std() + self.epsilon)
            img_yolo[mask == 0] = 0

        return img_yolo, seg_yolo, pta, ape, at
