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
from tiatoolbox.models.engine.patch_predictor import(
    IOPatchPredictorConfig,
    PatchPredictor
)

#Torch-related
import torch
from torchvision import transforms

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


