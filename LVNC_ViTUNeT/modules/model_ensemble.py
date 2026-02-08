from typing import List, Type
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class ModelEnsembleBase(nn.Module):
    def __init__(self, models: List[nn.Module], weights:torch.Tensor=None, num_spatial_dim: int =2, device = None):
        """Base class to wrap different `nn.Modules` and combine their outputs ia a weighted sum 

        Args:
            models (List[nn.Module]): List of modules whose output will be combined
            weights (torch.Tensor, optional): Weights used for the weighted sum of the outputs of the different models.
                                              If `None`, the mean of the outputs is computed. Defaults to None.
            num_spatial_dim (int, optional): Number of spatial dimensions. 2 for 2D problems, 3 for 3D... Defaults to 2.
            device ([type], optional): Device. Defaults to None.
        """
        super().__init__()
        if not isinstance(models, nn.ModuleList):
            models = nn.ModuleList(models)
        self.models = models
        if weights is None:
            self.weights = torch.empty(size=(len(self.models),), device=device).fill_(1/len(self.models))
        else:
            self.weights = weights
        assert len(self.weights)==len(self.models)
        if self.weights.ndim==1:
            self.weights =  self.weights[(None,)*(1+1+num_spatial_dim)].transpose(-1,0)       

    def forward(self, x):
        # TODO: softmax necesario para el caso de la segmentación. Parametrizar
        outputs = torch.stack([F.softmax(m(x), dim=1) for m in self.models]) # Shape MxNxCx(spatial dimensions)
        return torch.sum(self.weights*outputs, axis=0) # Importante el axis para que la `shape` de la salida coincida con la de cada modelo invididual.

class PLModuleEnsemble(ModelEnsembleBase):
    def __init__(self, pl_module_classes: List[Type[pl.LightningModule]], pl_module_arguments: List[dict], checkpoints_paths: List[Path], weights: torch.Tensor, num_spatial_dim: int, device):
        assert len(pl_module_classes) == len(pl_module_arguments) == len(checkpoints_paths)
        models = [
            cl.load_from_checkpoint(chck, map_location=device, **kwargs)
            for cl, kwargs, chck in zip(pl_module_classes, pl_module_arguments, checkpoints_paths)
        ]
        super().__init__(models, weights=weights, num_spatial_dim=num_spatial_dim, device=device)


class PlCommonModuleEnsemble(PLModuleEnsemble):
    def __init__(self, pl_module_class: Type[pl.LightningModule], pl_module_arguments: dict, checkpoints_paths: List[Path], weights: torch.Tensor=None, num_spatial_dim: int=2, device=None):
        super().__init__([pl_module_class]*len(checkpoints_paths), [pl_module_arguments]*len(checkpoints_paths), checkpoints_paths, weights, num_spatial_dim, device)