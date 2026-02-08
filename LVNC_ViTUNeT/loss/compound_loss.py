import torch
import torch.nn as nn

from modules.helper import RobustIdentity

class CompoundLoss(nn.Module):
    def __init__(self, losses):
        super(CompoundLoss, self).__init__()
        def build_loss(l):
            return l["function"](**l.get("params", {}))
        self.weights = torch.Tensor([l.get("comp_weight", 1) for l in losses])
        self.losses = nn.ModuleList([build_loss(l) for l in losses])
        self.non_linearities = [l.get("non_linearity", RobustIdentity()) for l in losses]
        self.channel_dim = [l.get("channel_dim", False) for l in losses]
        self.any_channel_dim = any(self.channel_dim)

    def forward(self, outputs, target):
        if self.any_channel_dim:
            target_cd = torch.unsqueeze(target, 1)

        loss = 0
        for w, l, nl, cd in zip(self.weights, self.losses, self.non_linearities, self.channel_dim):
            loss+= w*l(nl(outputs), target_cd if cd else target)
        
        return loss