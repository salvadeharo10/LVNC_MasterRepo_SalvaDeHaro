#%%
import os
from pydicom import dcmread
import pickle
import numpy as np

import matplotlib.pyplot as plt

from data.preprocessing import Preprocessor2D

#%%
preprocessor = Preprocessor2D(target_size=(512,512), use_mask=False) # No podemos usar mask porque en inferencia no la tendremos

#%%
raw_folder = "../LVNC_dataset/raw_dataset"
patient="P150"
slice=6
dcm = dcmread(os.path.join(raw_folder, f"dicom/{patient}_{slice}.dcm"))
img = dcm.pixel_array
gt = pickle.load(open(os.path.join(raw_folder, f"segmentation/{patient}_{slice}.pick"), "rb"))
# %%
img_p, gt_p, _ = preprocessor(img, gt)
# %% Zero mean
img_p.mean()
# %% Unit variance
img_p.std()
# %% 
img_p_p = (img_p*img.std())+img.mean()
# %%
img.dtype
# %%
img_p_p.dtype
# %%
plt.imshow(img)
# %%
plt.imshow(img_p)
# %%
plt.imshow(img_p_p)

# %%

# %%
