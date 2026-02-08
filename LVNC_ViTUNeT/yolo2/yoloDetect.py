def recortar_mascara(img, mask, x1, x2, y1, y2):
    #puede que la mascara y la imagen sobre la que se ha sacado el bounding box tengan dimensiones distintas
    
    input_height, input_weight, _ = img.shape
    mask_height, mask_weight = mask.shape

    factor_escala_y = mask_height/input_height
    factor_escala_x = mask_weight/input_weight

    x1 = int(x1*factor_escala_x)
    y1 = int(y1*factor_escala_y)
    x2 = int(x2*factor_escala_x)
    y2 = int(y2*factor_escala_y)

    mask_out = mask[y1:y2, x1:x2]

    return mask_out


def process_image_and_detect_with_yolo(yolo_model, img, seg):
    #sin paralelismo: pred = yolo_model.predict(img, verbose = False)[0]
    pred = yolo_model.predict(img, verbose = False, device = "cpu")[0] #con paralelismo

    if len(pred.boxes) == 0: #el modelo YOLO no encuentra nada
        return None, None
    else:
        bbox = pred.boxes.xyxy[0]
        #extraer confianza de prediccion
        conf = pred.boxes.conf[0].item()
        # Extraer las coordenadas del bounding box
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Obtenemos coordenadas del centro de la ROI en el sistema referencia de la imagen
        center_of_detection = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        #radio de la ROI ajustado en base a la confianza, queremos asegurar que se coge todo el LV
        radio = max((x2 - x1), (y2 - y1)) / 1.25
        radio_ajustado = int(radio * (1 + (1 - conf)))
        if radio_ajustado >= center_of_detection[0] or radio_ajustado >= center_of_detection[1]:
            return None, None
        #ajustar el margen de confianza del roi en base a la exactitud de YOLO
        x1 = center_of_detection[0] - radio_ajustado
        y1 = center_of_detection[1] - radio_ajustado
        x2 = center_of_detection[0] + radio_ajustado
        y2 = center_of_detection[1] + radio_ajustado

        img_output = img[y1:y2, x1:x2] # es una ROI
        
        seg_output = recortar_mascara(img, seg, x1, x2, y1, y2)

        return img_output, seg_output
    
