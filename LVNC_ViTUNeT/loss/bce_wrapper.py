import torch
from torch.nn import BCEWithLogitsLoss

class BCEWithLogitsWrapper(BCEWithLogitsLoss):
    """
    Converts the target to onehot encoding before computing torch.nn.BCEWithLogitsLoss
    """    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim==3:
            target = torch.unsqueeze(target, 1)

        num_classes = input.size(1)
        
        target_onehot = torch.FloatTensor(target.size(0),num_classes, target.size(2), target.size(3)).zero_()#.to(target.device)
        target_onehot.scatter_(1, target, 1)
        
        return super().forward(input, target_onehot)