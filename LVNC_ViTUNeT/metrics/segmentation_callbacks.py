from genericpath import exists
import os

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from pydicom import dcmread
from pytorch_lightning.callbacks import Callback

import torch

from palettable.cartocolors import qualitative

from data.maps_utils import image_from_onehot, logits_to_onehot, compute_pta
from data.utils import dicom_to_image
from metrics.dice import compute_dice
from metrics.pta_difference import compute_pta_difference
import pickle

import glob

class StoreTestSegmentationResults(Callback):
    def __init__(self, output_folder, classes, test_slice_list, output_size=(800, 800), palette=None, save_images=True, raw_folder=None, preproc_folder = None, group_results_by = None, exclude_bg=True, compute_volume_diff=False):
        super().__init__()
        self.output_folder = output_folder
        if not os.path.isdir(output_folder):
            Path(output_folder).mkdir(parents=True)
        self.classes = classes
        self.includes_ic = "ic" in classes
        self.test_slice_list = test_slice_list
        self.output_size = output_size
        self.final_test_metrics = []
        self.save_images = save_images
        self.raw_folder = raw_folder
        self.preproc_folder = preproc_folder
        if palette is None:
            palette = qualitative.get_map(f"Bold_{len(classes)}").colors
        self.palette=palette
        self.group_results_by = group_results_by
        self.compute_volume_diff = compute_volume_diff
        if group_results_by :
            self.df_raw = pd.read_pickle(os.path.join(self.raw_folder, "df_info.pick"))
        if compute_volume_diff:
            self.df_raw_pat = pd.read_pickle(os.path.join(self.raw_folder, "df_info_patients.pick"))
            self.computed_areas = []

        self.mean_dice = None
        self.exclude_bg = exclude_bg # If class 0 should be excluded from dice computation
    
    def on_test_batch_end(self, trainer, module, output, batch, batch_idx):
        epoch = trainer.current_epoch
        step = trainer.global_step

        original_idx = output["original_idx"]
        target = output["target"].detach().cpu()
        pred = output["logits"].detach().cpu()
        pred_onehot, target_onehot = logits_to_onehot(pred, target, len(self.classes))
        dice = compute_dice(pred_onehot, target_onehot, num_classes=len(self.classes))
        pta_difference = compute_pta_difference(pred, target, self.includes_ic)
        if self.compute_volume_diff:
            # We assume that for the sime study (patient) pixel spacing, thickness and interslice spacing does not change. 
            # Thus, those factors will not influence the final VT%% after the division
            _, APE, AT = compute_pta(torch.argmax(pred, axis=1), rh=1, rv=1, class_TI=3 if self.includes_ic else 2)
        for i in range(len(original_idx)):
            pat, slc = self.test_slice_list[original_idx[i]]

            new_dict = {
                "patient": pat,
                "slice": slc,
                "pta_difference": pta_difference[i].item() if pta_difference.dim() > 0 else pta_difference.item()
            }
            new_dict.update({f"dice_{c}": dice[i][j].item() for j, c in enumerate(self.classes) if j!=0 or not self.exclude_bg})
            new_dict.update({f"avg_dice": dice[i][1 if self.exclude_bg else 0:].mean().item()})
            self.final_test_metrics.append(new_dict)

            if self.compute_volume_diff:
                self.computed_areas.append({
                    "patient": pat,
                    "slice": slice,
                    "APE": APE[i].item() if APE.dim() > 0 else APE.item(),
                    "AT": AT[i].item() if AT.dim() > 0 else AT.item()
                })

            if self.save_images:
                # Save segmentation output
                segmentation = image_from_onehot(pred_onehot[i], self.palette, skip_bg=True)

                #try:
                 #   dcm_path = os.path.join(self.raw_folder, "dicom", f"{pat}_slc.dcm")
                  #  dcm = dcmread(dcm_path)
                   # dcm_img = dicom_to_image(dcm.pixel_arry, self.output_size)
                    #back = Image.fromarray(dcm_img).convert("RGBA")
                #except:
                 #   original_image_path = os.path.join(self.raw_folder, "jpg", f"{pat}_{slc}.jpg")
                  #  back = Image.open(original_image_path).convert('RGBA').resize(self.output_size)
                
                img_path = os.path.join(self.preproc_folder, "images", f'{pat}_{slc}.pick')
                img = pickle.load(open(img_path, 'rb'))
                back = Image.fromarray(img).convert('RGBA')

                
                front = Image.fromarray(segmentation).convert('RGBA').resize(self.output_size)
                front.putalpha(50) # Half alpha
                data = np.array(front)

                r1, g1, b1,a1 = 0, 0, 0,50 # Original value
                r2, g2, b2,a2 = 0, 0, 0,0 # Value that we want to replace it with

                red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
                mask = (red == r1) & (green == g1) & (blue == b1) & (alpha==a1)
                data[:,:,:4][mask] = [r2, g2, b2,a2]

                front = Image.fromarray(data)
                Image.alpha_composite(back, front).save(os.path.join(self.output_folder, f'{pat}_{slc:02d}_out.png'))

    def on_test_epoch_end(self, *args):
        final_dict ={"per_slice": self.final_test_metrics, "aggregation": {}}
        df = pd.DataFrame(self.final_test_metrics)
        reductions = ["mean", "std"] # It must be an Iterable
        names = ["pta_difference"] + [f"dice_{c}" for j, c in enumerate(self.classes) if j!=0 or not self.exclude_bg] +["avg_dice"]

        if self.compute_volume_diff:
            df_areas = pd.DataFrame(self.computed_areas)
            df_vol = df_areas.groupby("patient")[["APE", "AT"]].agg("sum")
            df_vol["VT% computed"] =  (100*df_vol["AT"]/(df_vol["AT"]+df_vol["APE"])) #.apply(lambda x:x.item())
            df_vol["VT% computed"] = df_vol["VT% computed"].fillna(0)
            df_vol = df_vol.merge(self.df_raw_pat, on="patient")
            df_vol["VT% diff"] = abs(df_vol["VT%"]-df_vol["VT% computed"])
            df_vol.sort_values(by="patient", inplace=True)
            self.volume_dict = {
                "per_patient": df_vol[["patient", "VT%", "VT% computed", "VT% diff"]].to_dict("records"),
            }
            if self.group_results_by:
                df_agg = df_vol.groupby(self.group_results_by)["VT% diff"].agg(reductions) # https://stackoverflow.com/questions/60229375/solution-for-specificationerror-nested-renamer-is-not-supported-while-agg-alo
                self.volume_dict["aggregation_vt_diff"] = df_agg.to_dict("index")
            else:
                df_agg = df_vol.agg({"VT% diff": reductions})
                self.volume_dict["aggregation_vt_diff"] = df_agg.to_dict()

            with open(os.path.join(self.output_folder, "_result_volumes.json"), "w") as f:
                json.dump(self.volume_dict, f, indent=4)        

        for reduction in reductions:
            if self.group_results_by:
                df_m = pd.merge(df, self.df_raw, on=["patient", "slice"])
                df_agg = df_m.groupby(self.group_results_by)[names].agg(reduction) # https://stackoverflow.com/questions/60229375/solution-for-specificationerror-nested-renamer-is-not-supported-while-agg-alo
                final_dict["aggregation"][reduction] = df_agg.to_dict("index")
            else:
                df_agg = df[names].agg(reduction)
                final_dict["aggregation"][reduction] = df_agg.to_dict()
        
        # Mean dice accross elements and classes
        self.mean_dice = df["avg_dice"].mean(axis=0)
        self.summary_results = final_dict["aggregation"]
        
        # Log to Lightning module loggers
#        module = args[1]
#        module.log_dict({f"val_final/{k}": v for  k,v in final_dict["aggregation"].items()})

        with open(os.path.join(self.output_folder, "_result.json"), "w") as f:
            json.dump(final_dict, f, indent=4)
