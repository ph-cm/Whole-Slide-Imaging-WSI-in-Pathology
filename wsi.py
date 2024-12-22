from __future__ import annotations

import logging 
import warnings
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

#Downloading data and files
import shutil
from pathlib import Path
from zipfile import ZipFile

#Data processing and visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import PIL
import contextlib
import io
from sklearn.metrics import accuracy_score, confusion_matrix

#TIAToolbox for WSI loading and processing
from tiatoolbox import logger
from tiatoolbox.models.architecture import vanilla
from tiatoolbox.utils.misc import download_data

from tiatoolbox.models.engine.patch_predictor import(
    IOPatchPredictorConfig,
    PatchPredictor
)

#Torch-related
import torch
from torchvision import transforms

import os
import openslide

#Configure plotting 
mpl.rcParams["figure.dpi"] = 160 #for high resolution
mpl.rcParams["figure.facecolor"] = "white" #text is visible

#if its not using GPU, change ON_GPU to false
ON_GPU = True

#Function to supress console output for overly verbose code blocks
def supress_console_output():
    return contextlib.redirect_stderr(io.StringIO())


#CLean-up before a run
warnings.filterwarnings("ignore")
global_save_dir = Path("./tmp/")

def rmdir(dir_path: str | Path) -> None:
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)
        
rmdir(global_save_dir)
global_save_dir.mkdir()
logger.info("Creating new directory %s", global_save_dir)

#Downloading the data
wsi_path = global_save_dir / "sample_wsi.svs"
patches_path = global_save_dir / "kather100k-validation-sample.zip"
weights_path = global_save_dir / "resnet18-kather100k.pth"

logger.info("Download has started. Please wait. . .")

#Downloading and unzip a sample whole-slide image
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs",
    wsi_path,
)

#Downloading and unzip a sample of the validation set used to train the kather 100k dataset
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/datasets/kather100k-validation-sample.zip",
    patches_path,
)
with ZipFile(patches_path, "r") as zipfile:
    zipfile.extractall(path=global_save_dir)
    
#Download pretrained model weights for WSI classification using ResNet18 architecture
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth",
    weights_path,
)

logger.info("Download is complete.")

#Reading data
#Read the patch data and create a list of patches and a list of corresponding labels
import os
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the global save directory and dataset path
global_save_dir = Path("C:/Users/phenr/Desktop/python-ws/Whole Slide Imaging/tmp")
dataset_path = global_save_dir / "kather100k-validation-sample"

# Set the file extension for images
image_ext = ".tif"

# Label dictionary mapping class names to IDs
label_dict = {
    "BACK": 0,  # Background (empty glass region)
    "NORM": 1,  # Normal colon mucosa
    "DEB": 2,   # Debris
    "TUM": 3,   # Colorectal adenocarcinoma epithelium
    "ADI": 4,   # Adipose
    "MUC": 5,   # Mucus
    "MUS": 6,   # Smooth muscle
    "STR": 7,   # Cancer-associated stroma
    "LYM": 8,   # Lymphocytes
}

class_names = list(label_dict.keys())
class_labels = list(label_dict.values())

# Function to grab files from a directory
def grab_files_from_dir(directory, file_types="*"):
    """
    Retrieves file paths from the specified directory that match the given file type(s).
    """
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory {directory} does not exist or is not a valid directory.")
    return [str(file) for file in directory_path.rglob(file_types) if file.is_file()]

# Ensure the dataset path exists
if not dataset_path.exists():
    raise FileNotFoundError(f"The dataset directory {dataset_path} does not exist. Please check the path.")

# Generate lists of patches and their corresponding labels
patch_list = []
label_list = []

for class_name, label in label_dict.items():
    dataset_class_path = dataset_path / class_name
    if not dataset_class_path.exists():
        logger.warning(f"Directory {dataset_class_path} does not exist. Skipping class: {class_name}")
        continue
    # Retrieve files for the class
    patch_list_single_class = grab_files_from_dir(
        dataset_class_path,
        file_types="*" + image_ext
    )
    patch_list.extend(patch_list_single_class)
    label_list.extend([label] * len(patch_list_single_class))

# Display dataset statistics
plt.figure(figsize=(10, 6))
plt.bar(class_names, [label_list.count(label) for label in class_labels], color='skyblue')
plt.title("Dataset Statistics")
plt.xlabel("Patch Type")
plt.ylabel("Number of Patches")
plt.xticks(rotation=45)
plt.show()

# Log dataset statistics
for class_name, label in label_dict.items():
    logger.info(
        "Class ID: %d -- Class Name: %s -- Number of images: %d",
        label,
        class_name,
        label_list.count(label)
    )

# Log the total number of patches
logger.info("Total number of patches: %d", len(patch_list))
