import json
import numpy as np
import cv2

def process_contours(data):
    size = data["size"]
    height, width = [int(d) for d in size.split("x")]

    # Contour of external wall
    strCPE = data["CPE"]
    strCPEsplit = strCPE.split("#")
    CPEpoints = []

    for substr in strCPEsplit:
        if substr != "":
            CPEpoints.append([int(d) for d in substr.split(",")])

    CPEpoints = np.array([[np.array(xi) for xi in CPEpoints]])

    # Contour of external trabecular wall
    strCTDE = data["CTDE"]
    strCTDEsplit = strCTDE.split("#")
    CTDEpoints = []

    for cadena in strCTDEsplit:
        if cadena != "":
            cadenaSplit = cadena.split(",")
            x = cadenaSplit[0]
            y = cadenaSplit[1]
            CTDEpoints.append([int(x), int(y)])

    CTDEpoints = np.array([[np.array(xi) for xi in CTDEpoints]])

    # Contours of inner trabeculae
    CTIcontours = []
    CTIpoints = []
    for contour in data["CTI"]:
        strCTIsplit = contour.split("#")
        points = []
        img = np.zeros(shape=(height, width), dtype=np.uint8)
        for cadena in strCTIsplit:
            if cadena != "":
                cadenaSplit = cadena.split(",")
                x = cadenaSplit[0]
                y = cadenaSplit[1]
                img[int(y), int(x)] = 255
                points.append([int(x), int(y)])
        CTIpoints.append(points)
        # https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
        contours_aux = [
            np.array(c).reshape((c.shape[0], c.shape[2])).astype("int")
            for c in contours
        ]

        CTIcontours.extend(contours_aux)

    CTIcontours = [CTIcontours]
    CTIpoints = np.array([[np.array(xi) for xi in CTIpoints]])
    CTIcontours_np = np.array([np.array(xi) for xi in CTIcontours])#, dtype=object)

    return CPEpoints, CTDEpoints, CTIcontours_np, CTIpoints

def class_mask_from_contour(file_path: str, desired_output=None, debug = False, show=False):
    # Load contour file
    with open(file_path) as json_file:
        contour_data = json.load(json_file)
    
    height, width = [int(d) for d in contour_data["size"].split("x")]
    if debug:
        print(f"Size: {height}x{width}")

    (
        CPEpoints,
        CTDEpoints,
        CTIcontours,
        CTIpoints,
    ) = process_contours(contour_data)
    
    # img = np.zeros((height, width, 3), np.uint8)
    classes = np.zeros((4, height, width), np.uint8)

    # cv2.drawContours(img, CPEpoints, -1, (255), -1)
    cv2.drawContours(classes[1], CPEpoints, -1, 1,
                     -1)  # Máscara de clases

    # cv2.drawContours(img, CTDEpoints, -1, (0, 255, 0), -1)
    cv2.drawContours(classes[2], CTDEpoints, -1, 1, -1)  # Máscara de clases

    for contour in CTIcontours:
        # cv2.drawContours(img, contour, -1, (255, 100, 255), thickness=1)
        if len(contour) == 0:
            continue
        if len(np.shape(contour[0])) == 1:
            contour = [contour]
        cv2.drawContours(classes[3], contour, -1, 1, -1)

    classes[0][np.argmax(classes, axis=0)==0] = 1

    if desired_output is None:
        if show:
            cv2.imshow("Image from mask", image_from_mask(classes))
            cv2.waitKey(0)
        return classes

    classesResized = resize_mask(classes, target_size=desired_output)

    if show:
        cv2.imshow("Image from mask", image_from_mask(classes))
        cv2.imshow("Image from resized", image_from_mask(classesResized))
        cv2.waitKey(0)
    return classes