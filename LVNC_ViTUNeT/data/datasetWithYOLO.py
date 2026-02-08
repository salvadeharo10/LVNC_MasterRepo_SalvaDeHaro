import os
import numpy as np
import torch
import pickle
from PIL import Image
from yolo.yoloDetect import process_image_and_detect_with_yolo
import cv2
from yolo.utils import recortar_ampliar_normalizar_imagen
from data.maps_utils import resize_class_map

class LVNCDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_folder, preproc_folder, list_slices, yolo_model, transform=None):
        self.raw_data_folder = raw_data_folder
        self.preproc_folder = preproc_folder
        self.list_slices = list_slices
        self.transform = transform
        self.yolo_model = yolo_model
    
    def __len__(self):
        return len(self.list_slices)

    def __getitem__(self, index):
        patient, num_slice = self.list_slices[index]
        img_jpg_path = os.path.join(self.raw_data_folder, f'jpg/{patient}_{num_slice}.jpg')
        img_jpg = np.array(Image.open(img_jpg_path))

        seg = pickle.load(open(os.path.join(self.preproc_folder, f'gt/{patient}_{num_slice}.pick'), 'rb'))
    
        img_yolo, seg_yolo = process_image_and_detect_with_yolo(self.yolo_model, img_jpg, img_jpg_path, seg)

        if img_yolo is None: #si el modelo YOLO ha fallado...
            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Encontrar el contorno más grande (asumiendo que es el ventrículo izquierdo)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour) #(x,y) vertice superior izquierdo
            x2 = x + w
            y2 = y + h
            img_yolo = recortar_ampliar_normalizar_imagen(img_jpg, x, x2, y, y2)
            seg_yolo = seg[int(y):int(y2), int(x):int(x2)]
            seg_yolo = resize_class_map(seg_yolo, target_size=seg.shape, order = 3)
        
        
        s = img_yolo.shape

        if self.transform:
            transformed = self.transform(image = img_yolo, mask = seg_yolo)
            img_yolo = transformed["image"]
            seg_yolo = transformed["mask"]

        assert s==img_yolo.shape

        return {
            "patient": patient,
            "num_slice": num_slice,
            "idx": index,
            "image": np.expand_dims(img_yolo, axis=0).astype(np.float32),
            "mask": seg_yolo.astype(np.int_) # Long
        }

def get_classes_proportion(dataset: torch.utils.data.Dataset, num_classes: int, batch_size:int =32, num_workers: int = 4):
    total = torch.zeros(num_classes, dtype=torch.long)
    for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        masks=batch["mask"]
        uq = masks.unique(sorted=True, return_counts=True)
        total[uq[0]] += uq[1]
    
    return total/sum(total)