import numpy as np
from skimage import color
from skimage.transform import resize
import cv2 as cv

from .maps_utils import resize_class_map, compute_pta, compute_area_ic
from .utils import s16_to_u8, pipeline_preprocessing_mri

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


class Preprocessor2D(object):
    def __init__(self,target_size, normalization_method="zscore", rm_ic=False, multi_label = False, use_mask=False, compute_pta = False, compute_area_ic = False, epsilon=1e-8, nearest_neigh_mask=False, combined_resize=False, proc_mri_image = False):
        self.normalization_method = normalization_method
        self.rm_ic = rm_ic
        self.multi_label = multi_label
        self.use_mask = use_mask
        self.target_size = target_size
        self.compute_pta = compute_pta
        self.compute_area_ic = compute_area_ic
        self.epsilon = epsilon
        self.nn_mask = nearest_neigh_mask
        self.combined_resize = combined_resize # If true, ignore nn_mask
        self.proc_mri_image = proc_mri_image


    def __call__(self, dicom_array, segmentation):
        if self.combined_resize:
            segmentation_r = resize_class_map(segmentation, order=3, target_size=self.target_size)
            segmentation_nn = resize(segmentation.astype(float), self.target_size, order=0, mode="edge", anti_aliasing=False)
            mask=np.logical_and(segmentation_r==0, np.logical_or(segmentation_nn==2, segmentation_nn==3))
            segmentation_r[mask] = segmentation_nn[mask] # Replace "black points" with the value of the NN resize if it contains interval cavity or trabeculae
            segmentation = segmentation_r
        else:
            segmentation = resize_class_map(segmentation, target_size=self.target_size, order = 0 if self.nn_mask else 3)
            
        if self.compute_pta:
            pta, ape, at = compute_pta(segmentation, rh=1, rv=1)
        else:
            pta = None
            ape = None
            at = None
        
        if self.compute_area_ic and not self.rm_ic:
            aic = compute_area_ic(segmentation, rh = 1, rv = 1)
        else:
            aic = None

        if self.multi_label:
            # One class containing the external wall and the trabeculae
            # Another class with the trabeculae
            binary_masks = np.zeros((2 if self.rm_ic else 3, *self.target_size), np.uint8)
            binary_masks[0, segmentation==1] = 1
            binary_masks[0, segmentation==3] = 1
            if self.rm_ic:
                binary_masks[1, segmentation==3] = 1
            else:
                binary_masks[1, segmentation==2] = 1
                binary_masks[2, segmentation==3] = 1

            segmentation = binary_masks
        else:
            if self.rm_ic:
                segmentation = remove_classes(segmentation, [2])

        image = s16_to_u8(dicom_array)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY) #(255*color.rgb2gray(image)).astype(np.uint8)
        if image.shape!=self.target_size:
            image = resize(image, self.target_size, order=3, anti_aliasing=False, preserve_range=True)
        if self.proc_mri_image:
            image = pipeline_preprocessing_mri(image, target_size = self.target_size)
        if self.normalization_method=="zscore":
            if self.use_mask:
                mask = segmentation > 0
            else:
                mask = np.ones(segmentation.shape, dtype=bool)
            image = (image - image.mean())/(image.std() + self.epsilon)
            image[mask == 0] = 0

        return image, segmentation, pta, ape, at, aic
