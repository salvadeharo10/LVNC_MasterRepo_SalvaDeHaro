from skimage.transform import resize
from skimage import color
import numpy as np
from data.maps_utils import resize_class_map
from data.utils import s16_to_u8

def process_image_and_detect_with_yolo(yolo_model, img, img_path, seg):
    #YOLO esta entrenado con imagenes en 800x800x3
    resized_img = resize(img, (800, 800), mode='constant', anti_aliasing=True)
    resized_img = (resized_img * 255).astype(np.uint8)

    pred = yolo_model.predict(resized_img, verbose = False)[0]

    if len(pred.boxes.xyxy) == 0:
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
        print(f'imagen problema: {img_path}')
    else:
        bbox = pred.boxes.xyxy[0]
        # Extraer las coordenadas del bounding box
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        #preparar la imagen para entrenar a vitunet
        img = s16_to_u8(img)
        if len(img.shape) == 3:
            img = (255*color.rgb2gray(img)).astype(np.uint8)
        if img.shape!=seg.shape:
            img = resize(img, seg.shape, order=3, anti_aliasing=False, preserve_range=True)

        #normalización z-score
        img_output = (img - img.mean())/(img.std() + 1e-8)

        # Crear una máscara para los píxeles fuera del bounding box
        mask = np.zeros_like(img)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1

        # Aplicar la máscara para hacer los píxeles fuera del ROI negros
        img_output *= mask
           

        return img_output, seg
    return None, None