from pathlib import Path
import json

"""
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
"""
import albumentations as A

"""
Isensee Fabian, Jäger Paul, Wasserthal Jakob, Zimmerer David, Petersen Jens, Kohl Simon, 
Schock Justus, Klein Andre, Roß Tobias, Wirkert Sebastian, Neher Peter, Dinkelacker Stefan, 
Köhler Gregor, Maier-Hein Klaus (2020). batchgenerators - a python framework for data 
augmentation. doi:10.5281/zenodo.3632567
"""


def save_augment_config(filepath: Path, framework: str, transform):
    with open(filepath, 'w') as f:
        if framework=="albumentations":
            t_dict ={
                'framework': framework,
                'pipeline': A.to_dict(transform)
            } 
            json.dump(t_dict, f)

def parse_augment_config(filepath: Path):
    with open(filepath, "r") as f:
        config = json.load(f)
    
    framework = config["framework"]
    pipeline = config["pipeline"]
    
    if framework=="albumentations":
        transforms =  A.from_dict(pipeline)

    return transforms