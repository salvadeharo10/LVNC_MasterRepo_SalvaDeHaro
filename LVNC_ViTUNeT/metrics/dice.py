from typing import List

import torch
from torchmetrics import Metric, MetricCollection

from data.maps_utils import logits_to_onehot

def compute_dice(output: torch.Tensor, target: torch.Tensor, num_classes: int = None, compute_classes: List[int] = None, smooth=1e-8):
    """
    output and target must be onehot representations
        NxCxHxW
    """
    if compute_classes is None:
        compute_classes = list(range(num_classes))

    output = output[:,compute_classes, :, :]
    target = target[:,compute_classes, :, :]

    intersection = torch.mul(output, target)
    dice = (2*torch.sum(intersection, dim=(2,3))+smooth)/(torch.sum(output, dim=(2,3)) + torch.sum(target, dim=(2,3)) +smooth)    

    return dice


class MultiClassDiceCollection(MetricCollection):
    def __init__(self, classes, aggregate_mean = True, smooth=1e-8, prefix=""):
        super().__init__(
            {f"{prefix}dice_{c}": SingleClassDiceMetric(len(classes), i, aggregate_mean=aggregate_mean, smooth=smooth) for i, c in enumerate(classes)}
        )

class SingleClassDiceMetric(Metric):
    def __init__(self, num_classes: int, compute_class: int, aggregate_mean = True, smooth=1e-8):
        super().__init__()
        self.compute_class = compute_class
        self.num_classes = num_classes
        self.aggregate_mean = aggregate_mean
        self.smooth = smooth
        self.add_state("dice", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        Expected onehot representations (produced by `logits_to_onehot`)
        """

        dice = compute_dice(output, target, compute_classes=[self.compute_class], smooth=self.smooth)

        self.dice = torch.cat([self.dice, torch.squeeze(dice, 1)])

    def compute(self):
        if self.aggregate_mean:
            return torch.mean(self.dice)
        else:
            return self.dice

