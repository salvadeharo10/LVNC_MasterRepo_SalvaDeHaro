import os
import pickle
import numpy as np
import pandas as pd
import shutil
import cv2
import torch
import skimage.exposure as exposure
from skimage.transform import resize
from ultralytics import YOLO



def calcular_area_bbox(yolo_prediction, j):  
    xywh = yolo_prediction.boxes.xywh[j]
    x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
        
    return int(w * h)
    
    

def procesar_ficheros_con_yolo_batch(path_to_ficheros, device, batch_size=8):

    yolo_model = YOLO('./yolo2/best_LV_IC_yolov11s.pt')
    
    print(f'Se usara el dispositivo: {device}')
    
    yolo_model.to(device)

    resultados = []
    
 
    for i in range(0, len(path_to_ficheros), batch_size):
            
        batch_files = path_to_ficheros[i:i+batch_size]
        imagenes = []
        metadatos = []
        
        for path_fichero in batch_files:
            
            fichero = os.path.basename(path_fichero)
            fichero = os.path.splitext(fichero)[0]
            try:
                paciente, slice_ = fichero.split('_')
            except ValueError:
                print(f"Formato de fichero inesperado: {fichero}")
                continue

           
            img = cv2.cvtColor(cv2.imread(path_fichero), cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"No se pudo leer la imagen: {path_fichero}")
                continue
            
            
            imagenes.append(img)
            metadatos.append((paciente, slice_))
        
       
        if not imagenes:
            continue
        
        
        predicciones = yolo_model(imagenes, verbose=False, device = device)
        
        
        for pred, (paciente, slice_) in zip(predicciones, metadatos):
            
            confianza_lv, confianza_ic = 0, 0
            area_lv, area_ic = 0, 0
            
            if pred is not None and len(pred.boxes) > 0:
            
                for j, clase in enumerate(pred.boxes.cls):
                    clase = int(clase.item())

                    confianza = pred.boxes.conf[j].item()
                    area = calcular_area_bbox(pred, j)
                    if clase == 0:
                        confianza_lv = confianza
                        area_lv = area
                    else:
                        confianza_ic = confianza
                        area_ic = area
                        
               
            resultados.append({
                'patient': paciente,
                'slice': slice_,
                'confianza LV': confianza_lv,
                'area bbox LV': area_lv,
                'confianza IC': confianza_ic,
                'area bbox IC': area_ic
            })
            
        print(f"Procesados {i+len(batch_files)} de {len(path_to_ficheros)} ficheros")
    
    
    return pd.DataFrame(resultados)    
    
    

path_to_all_jpg = '../LVNC_dataset/new_raw_dataset/jpg_via'
path_to_ficheros_jpg = [os.path.join(path_to_all_jpg, fichero) for fichero in os.listdir(path_to_all_jpg)]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
df_kaka = procesar_ficheros_con_yolo_batch(path_to_ficheros_jpg, device)

df_kaka.to_csv('confIC_AreaBBOX.csv')
