"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

# Source: https://github.com/bermanmaxim/LovaszSoftmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None, apply_softmax=True):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.apply_softmax = apply_softmax

    def forward(self, output, target):
        if self.apply_softmax:
            probas = F.softmax(output, dim=1)
        else:
            probas = output
        return lovasz_softmax(probas, target, classes=self.classes, per_image=self.per_image, ignore=self.ignore)


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        losses = [lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                  for prob, lab in zip(probas, labels)]
        return torch.mean(torch.stack(losses))
    else:
        return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas.sum() * 0.0
    C = probas.size(1)
    losses = []
    class_list = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_list:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.mean(torch.stack(losses))


def flatten_probas(probas, labels, ignore=None):
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid_mask = (labels != ignore)
    return probas[valid_mask], labels[valid_mask]
