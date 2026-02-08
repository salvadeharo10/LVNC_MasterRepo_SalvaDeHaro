import torch
from torch.utils.benchmark import Timer

import torchvision

def just_concat(x, y):
    return torch.cat([x, y], dim=1)

def crop_and_concat(x, y):
    _, _, H, W = x.shape
    y   = torchvision.transforms.CenterCrop([H, W])(y)
    torch.cat([x, y], dim=1)


a = torch.randn(16,960,25,25, device='cuda')
c = torch.randn(16,960,25,25, device='cuda')
b = torch.randn(16,960,26,26, device='cuda')

t0 = Timer(
    stmt='crop_and_concat(a,b)',
    setup='from __main__ import crop_and_concat',
    globals={
        'a': a,
        'b': b
    }
)

t1 = Timer(
    stmt='just_concat(a,b)',
    setup='from __main__ import just_concat',
    globals={
        'a': a,
        'b': c
    }
)


print(t0.timeit(1000))
print(t1.timeit(1000))