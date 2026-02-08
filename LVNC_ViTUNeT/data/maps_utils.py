import numpy as np
from skimage.transform import resize

import torch

def map_from_binary_masks(masks):
    # Argmax but getting the index of the last occurence
    return len(masks) - np.argmax(masks[::-1], axis=0) - 1 


def resize_class_map(class_map, order=3, target_size=(256,256), anti_aliasing=False, binary_threshold=0.5):
    result = np.zeros(target_size, dtype=class_map.dtype)
    for c in np.unique(class_map):
        mask = class_map == c
        if mask.shape!=target_size:
            mask = resize(mask.astype(float), target_size, order=order, mode="edge", clip=True, anti_aliasing=anti_aliasing)
        result[mask >=binary_threshold] = c
    return result


# https://stackoverflow.com/questions/48987774/how-to-crop-a-numpy-2d-array-to-non-zero-values
def smallest_box(a):
    r = a.any(1)
    if r.any():
        m,n = a.shape
        c = a.any(0)
        out = a[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax()]
    else:
        out = np.empty((0,0),dtype=bool)
    return out

def compute_rh_rv(original_size, current_size, pixel_spacing):
    return (
        (pixel_spacing[0] * original_size[0]) / current_size[0],
        (pixel_spacing[1] * original_size[1]) / current_size[1],
    )
    

'''
La matriz de adquisición es la resolución de la máquina de resonancia, lo que pasa que el sistema puede hacer interpolación en un último paso
y almacenar la imagen en unas dimensiones distintas a las de la matriz de adquisición.

Pixel spacing es un parámetro dicom que es una tupla: pixel_spacing = (pixel_spacing_h (por filas, o altura), pixel_spacing_w (por columnas, o anchura))

Normalmente suele ser pixel_spacing_h = pixel_spacing_w porque las imagenes se suelen almacenar en dimensiones H x W con H = W.

Pero basicamente, pixel_spacing_h indica la separación en milimetros entre pixeles adyacentes por filas (altura), mientras que
pixel_spacing_w contieen lo mismo pero en separación por columnas (o anchura).

Por tanto, si la imagen almacenada tiene una resolución de H x W pixeles, en milimetros seria:

H_mm = H x pixel_spacing_h
W_mm = W x pixel_spacing_w

Y por tanto, si calculamos por ejemplo Area IC en pixeles, para pasarla a milimetros tendremos que usar el pixel_spacing.

Aquí, la imagen se está tratando en sus dimensiones originales, por tanto podemos poner siempre rh = 1 y rv = 1, ya que current_size = original_size.
El pixel spacing da informacion para la imagen en su tamaño original. Si procesamos una imagen que originalmente era 2566 x 256 y ahora queremos calcular
su area en mm en 800 x 800, necesitariamos saber que originalmente era 256 x256, ya que el pixel spacing que tenemos informa para cuando la imagen era 256 x256 (forma original).

'''

def compute_mm(rhorizontal, rvertical, area):
    return area * rhorizontal * rvertical

def compute_pta(class_map, rh=None, rv=None, original_size=None, current_size=None,
                pixel_spacing=None, one_hot=False, class_PE=1, class_TDE=2, class_TI=3, volumes=False):
    
    # Important: take into account that if we are just interested PTA, we can use whatever non-zero values for rh and rv as those factor will be cancelled in the division

    assert (rh and rv) or (original_size and current_size and pixel_spacing)
    
    if one_hot:
        class_map = map_from_binary_masks(class_map)
        
    if rh is None:
        rh, rv = compute_rh_rv(original_size, current_size, pixel_spacing)
    
    # We force NxC(1)xHxW
    if len(class_map.shape)==2:
        if isinstance(class_map, np.ndarray):
            class_map = np.expand_dims(class_map, axis=0)
        elif isinstance(class_map, torch.Tensor):
            class_map = torch.unsqueeze(class_map, axis=0)

    if len(class_map.shape)==3:
        if isinstance(class_map, np.ndarray):
            class_map = np.expand_dims(class_map, axis=1)
        elif isinstance(class_map, torch.Tensor):
            class_map = torch.unsqueeze(class_map, axis=1)

    if volumes:
        # Take into account that for volumes we assume just one patient and we aggregate all the results together
        # Get volume of every part
        area_PE = (class_map == class_PE).sum(axis=(0,2,3)).squeeze(0)
        #area_TDE = (class_map == class_TDE).sum()
        area_TI = (class_map == class_TI).sum(axis=(0,2,3)).squeeze(0) 
    else:
        # Get area of every part
        area_PE = (class_map == class_PE).sum(axis=(2,3)).squeeze(1)
        #area_TDE = (class_map == class_TDE).sum()
        area_TI = (class_map == class_TI).sum(axis=(2,3)).squeeze(1) 
    
    # Compute PTA
    APE = compute_mm(rh, rv, area_PE)
    AT = compute_mm(rh, rv, area_TI)
    PTA = 100*AT/(AT+APE)
    # If NaN (division by zero), return 0
    PTA[PTA!=PTA] = 0
    APE[APE!=APE] = 0
    AT[AT!=AT] = 0
    
    #hay que arreglar esto. cuando volume = False y class_map tiene forma (N (instancias), C (canales) = 1, H, W)
    #entonces PTA, APE y AT son arrays de N dimensiones (un valor de PTA, APE y AT por cada instancia)
    #ahora mismo, cuando N > 1, siempre devolvemos los valores de la primera instancia.
    #cuando volume = True da igual N, porque al final PTA, APE y AT siempre serán un array con un único valor (el agregado del volumen)
    #es decir, cuando volume = True, APE es el volumen de la pared exterior, AT es el volumen trabecular y PTA es el porcentaje de volumen trabecular

    #solucion 1 posible
    if len(PTA) == 1:
        return PTA[0], APE[0], AT[0]
    else:
        return PTA, APE, AT
        


def compute_area_ic(class_map, rh=None, rv=None, original_size=None, current_size=None,
                pixel_spacing=None, one_hot=False, class_PE=1, class_IC=2, class_T=3, volumes=False):

    # Important: take into account that if we are just interested PTA, we can use whatever non-zero values for rh and rv as those factor will be cancelled in the division

    assert (rh and rv) or (original_size and current_size and pixel_spacing)
    
    if one_hot:
        class_map = map_from_binary_masks(class_map)
        
    if rh is None:
        rh, rv = compute_rh_rv(original_size, current_size, pixel_spacing)
    
    # We force NxC(1)xHxW
    if len(class_map.shape)==2:
        if isinstance(class_map, np.ndarray):
            class_map = np.expand_dims(class_map, axis=0)
        elif isinstance(class_map, torch.Tensor):
            class_map = torch.unsqueeze(class_map, axis=0)

    if len(class_map.shape)==3:
        if isinstance(class_map, np.ndarray):
            class_map = np.expand_dims(class_map, axis=1)
        elif isinstance(class_map, torch.Tensor):
            class_map = torch.unsqueeze(class_map, axis=1)

    if volumes:
        # Take into account that for volumes we assume just one patient and we aggregate all the results together
        # Get volume of every part
        area_IC = (class_map == class_IC).sum(axis=(0,2,3)).squeeze(0)
    else:
        # Get area of every part
        area_IC = (class_map == class_IC).sum(axis=(2,3)).squeeze(1)

    
    # Compute Area IC
    AIC = compute_mm(rh, rv, area_IC)

    AIC[AIC!=AIC] = 0
    
    #hay que arreglar esto. cuando volume = False y class_map tiene forma (N (instancias), C (canales) = 1, H, W)
    #entonces PTA, APE y AT son arrays de N dimensiones (un valor de PTA, APE y AT por cada instancia)
    #ahora mismo, cuando N > 1, siempre devolvemos los valores de la primera instancia.
    #cuando volume = True da igual N, porque al final PTA, APE y AT siempre serán un array con un único valor (el agregado del volumen)
    #es decir, cuando volume = True, APE es el volumen de la pared exterior, AT es el volumen trabecular y PTA es el porcentaje de volumen trabecular

    #solucion 1 posible
    if len(AIC) == 1:
        return AIC[0]
    else:
        return AIC



def image_from_onehot(mask, color, skip_bg=True):
    num_classes, height, width = np.shape(mask)
    image = np.zeros((height, width, 3), np.uint8)
    if color is None:
        color = [
            list(np.random.choice(range(50, 256), size=3))
            for i in range(num_classes)
        ]
    
    for c in range(num_classes):
        if skip_bg and c==0:
            continue

        np.putmask(image[:, :, 0], mask[c], color[c][0])
        np.putmask(image[:, :, 1], mask[c], color[c][1])
        np.putmask(image[:, :, 2], mask[c], color[c][2])

    return image


def logits_to_onehot(output: torch.Tensor, target: torch.Tensor, num_classes: int, non_linearity = None):
    assert len(output)==len(target)

    if non_linearity is not None:
        output = non_linearity(output)

    if len(output.size())==4:
        assert output.size(1)==num_classes
        output = torch.argmax(output, axis=1)
    
    output = torch.unsqueeze(output, 1)
    
    if len(target.size())==3:
        target = torch.unsqueeze(target, 1)

    # Convert to onehot
    output_onehot =  torch.FloatTensor(output.size(0),num_classes, output.size(2), output.size(3)).zero_().to(output.device)
    output_onehot.scatter_(1, output, 1)
    target_onehot =  torch.FloatTensor(target.size(0), num_classes, target.size(2), target.size(3)).zero_().to(output.device)
    target_onehot.scatter_(1, target, 1)

    return output_onehot, target_onehot  