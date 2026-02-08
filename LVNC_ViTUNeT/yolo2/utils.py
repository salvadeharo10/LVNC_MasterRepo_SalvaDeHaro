from skimage.transform import resize
from skimage import color
from data.utils import s16_to_u8
import numpy as np

def recortar_ampliar_normalizar_imagen(imagen, x1, x2, y1, y2):
    #el bounding box se ha sacado sobre una imagen 800x800, pero la imagen original no tiene porque tener ese tamaño
    # Calcular el bounding box respecto a las dimensiones originales
    input_height, input_weight, _ =imagen.shape
    factor_escala_x = input_weight/800
    factor_escala_y = input_height/800

    x1 = int(x1*factor_escala_x)
    y1 = int(y1*factor_escala_y)
    x2 = int(x2*factor_escala_x)
    y2 = int(y2*factor_escala_y)

    imagen = s16_to_u8(imagen)
    roi = imagen[y1:y2, x1:x2]
    if len(roi.shape) == 3:
        roi = (255*color.rgb2gray(roi)).astype(np.uint8)
    
    roi = resize(roi, (800, 800), order=3, anti_aliasing=False, preserve_range=True)

    #normalización z-score
    img_output = (roi - roi.mean())/(roi.std() + 1e-8)

    return img_output