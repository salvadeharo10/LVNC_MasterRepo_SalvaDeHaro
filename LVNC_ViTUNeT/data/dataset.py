import os
import numpy as np
import torch
import pickle

class LVNCDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, list_slices, transform=None):
        self.data_folder = data_folder
        self.list_slices = list_slices
        self.transform = transform
    
    def __len__(self):
        return len(self.list_slices)

    def __getitem__(self, index):
        patient, num_slice = self.list_slices[index]
        img = pickle.load(open(os.path.join(self.data_folder, f'images/{patient}_{num_slice}.pick'), 'rb'))
        seg = pickle.load(open(os.path.join(self.data_folder, f'gt/{patient}_{num_slice}.pick'), 'rb'))
        
        s = img.shape

        if self.transform:
            transformed = self.transform(image = img, mask = seg)
            img = transformed["image"]
            seg = transformed["mask"]

        assert s==img.shape

        return {
            "patient": patient,
            "num_slice": num_slice,
            "idx": index,
            "image": np.expand_dims(img, axis=0).astype(np.float32),
            "mask": seg.astype(np.int_) # Long
        }

def get_classes_proportion(dataset: torch.utils.data.Dataset, num_classes: int, batch_size:int =32, num_workers: int = 4):
    total = torch.zeros(num_classes, dtype=torch.long)
    for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        masks=batch["mask"]
        uq = masks.unique(sorted=True, return_counts=True)
        total[uq[0]] += uq[1]
    
    return total/sum(total)