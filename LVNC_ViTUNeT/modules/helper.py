import torch
import torch.nn as nn

class RobustIdentity(nn.Module):
    """
    The constructor ignores all the params
    forward returns the input element and ignores all the params
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self,x, *args, **kwargs):
        return x