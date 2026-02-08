from typing import Union

import numpy as np
from torchmetrics import Metric

import torch
from data.maps_utils import compute_pta

def compute_pta_difference(output: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], includes_ic: bool, non_linearity = None):
    """
    output: NxCxHxW
    target: N(x1)xHxW
    """
    if non_linearity:
        output = non_linearity(output)
    class_TI = 3 if includes_ic else 2
    pta_output, _, _ = compute_pta(torch.argmax(output, axis=1), rh=1, rv=1, class_TI=class_TI)
    pta_target, _, _ = compute_pta(target, rh=1, rv=1, class_TI=class_TI) 
    return abs(pta_output-pta_target)

# Computes differences in percentage of trabecular volume
def compute_vt_difference(output: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], includes_ic: bool, non_linearity = None):
    """
    Here we expect the segmentations for one patient stacked in a Nx1xHxW way
    The function returns the VT% difference
    """
    if non_linearity:
        output = non_linearity(output)
    class_TI = 3 if includes_ic else 2
    vt_output, _, _ = compute_pta(torch.argmax(output, axis=1), rh=1, rv=1, class_TI=class_TI, volumes=True)
    vt_target, _, _ = compute_pta(target, rh=1, rv=1, class_TI=class_TI, volumes=True) 
    return abs(vt_output-vt_target)
    

class PTADifferenceMetric(Metric):
    def __init__(self, classes, aggregate_mean=True):
        super().__init__()
        self.includes_ic = "ic" in classes
        self.aggregate_mean=aggregate_mean
        self.add_state("pta_difference", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        result = compute_pta_difference(output, target, self.includes_ic)
        if result.dim() == 0:
            result = result.unsqueeze(0)
        self.pta_difference = torch.cat([self.pta_difference, result])

    def compute(self):
        if self.aggregate_mean:
            return torch.mean(self.pta_difference)
        else:
            return self.pta_difference
        
