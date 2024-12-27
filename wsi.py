"""Import modules required to run the Jupyter notebook."""

from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging



if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import accuracy_score, confusion_matrix

from tiatoolbox import logger
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

mpl.rcParams["figure.dpi"] = 160  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode

device = "cuda"  # Choose appropriate device

warnings.filterwarnings("ignore")
global_save_dir = Path("./tmp/")


def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)


rmdir(global_save_dir)  # remove  directory if it exists from previous runs
global_save_dir.mkdir()
logger.info("Creating new directory %s", global_save_dir)

img_file_name = global_save_dir / "sample_tile.png"
wsi_file_name = global_save_dir / "sample_wsi.svs"
patches_file_name = global_save_dir / "kather100k-validation-sample.zip"
imagenet_samples_name = global_save_dir / "imagenet_samples.zip"

logger.info("Download has started. Please wait...")

# Downloading sample image tile
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/CRC-Prim-HE-05_APPLICATION.tif",
    img_file_name,
)

# Downloading sample whole-slide image
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs",
    wsi_file_name,
)

# Download a sample of the validation set used to train the Kather 100K dataset
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/datasets/kather100k-validation-sample.zip",
    patches_file_name,
)
# Unzip it!
with ZipFile(patches_file_name, "r") as zipfile:
    zipfile.extractall(path=global_save_dir)

# Download some samples of imagenet to test the external models
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/imagenet_samples.zip",
    imagenet_samples_name,
)
# Unzip it!
with ZipFile(imagenet_samples_name, "r") as zipfile:
    zipfile.extractall(path=global_save_dir)

logger.info("Download is complete.")

# read the patch data and create a list of patches and a list of corresponding labels

dataset_path = global_save_dir / "kather100k-validation-sample"

# set the path to the dataset
image_ext = ".tif"  # file extension of each image

# obtain the mapping between the label ID and the class name
label_dict = {
    "BACK": 0,
    "NORM": 1,
    "DEB": 2,
    "TUM": 3,
    "ADI": 4,
    "MUC": 5,
    "MUS": 6,
    "STR": 7,
    "LYM": 8,
}
class_names = list(label_dict.keys())
class_labels = list(label_dict.values())

# generate a list of patches and generate the label from the filename
patch_list = []
label_list = []
for class_name, label in label_dict.items():
    dataset_class_path = dataset_path / class_name
    patch_list_single_class = grab_files_from_dir(
        dataset_class_path,
        file_types="*" + image_ext,
    )
    patch_list.extend(patch_list_single_class)
    label_list.extend([label] * len(patch_list_single_class))


# show some dataset statistics
plt.bar(class_names, [label_list.count(label) for label in class_labels])
plt.xlabel("Patch types")
plt.ylabel("Number of patches")
plt.show()

# count the number of examples per class
for class_name, label in label_dict.items():
    logger.info(
        "Class ID: %d -- Class Name: %s -- Number of images: %d",
        label,
        class_name,
        label_list.count(label),
    )


# overall dataset statistics
logger.info("Total number of patches: %d", (len(patch_list)))

predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)
output = predictor.predict(imgs=patch_list, mode="patch", device=device)

acc = accuracy_score(label_list, output["predictions"])
logger.info("Classification accuracy: %f", acc)

# Creating and visualizing the confusion matrix for patch classification results
conf = confusion_matrix(label_list, output["predictions"], normalize="true")
df_cm = pd.DataFrame(conf, index=class_names, columns=class_names)

# show confusion matrix
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt=".0%")
plt.show()
plt.close()

# reading and displaying a tile image
input_tile = imread(img_file_name)

plt.imshow(input_tile)
plt.axis("off")
plt.show()
plt.close()

logger.info(
    "Tile size is: (%d, %d, %d)",
    input_tile.shape[0],
    input_tile.shape[1],
    input_tile.shape[2],
)

rmdir(global_save_dir / "tile_predictions")
img_file_name = Path(img_file_name)

predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)
tile_output = predictor.predict(
    imgs=[img_file_name],
    mode="tile",
    merge_predictions=True,
    patch_input_shape=[224, 224],
    stride_shape=[224, 224],
    resolution=1,
    units="baseline",
    return_probabilities=True,
    save_dir=global_save_dir / "tile_predictions",
    device=device,
)

# individual patch predictions sampled from the image tile

# extract the information from output dictionary
coordinates = tile_output[0]["coordinates"]
predictions = tile_output[0]["predictions"]

# select 4 random indices (patches)
rng = np.random.default_rng()  # Numpy Random Generator
random_idx = rng.integers(0, len(predictions), (4,))

for i, idx in enumerate(random_idx):
    this_coord = coordinates[idx]
    this_prediction = predictions[idx]
    this_class = class_names[this_prediction]

    this_patch = input_tile[
        this_coord[1] : this_coord[3],
        this_coord[0] : this_coord[2],
    ]
    plt.subplot(2, 2, i + 1), plt.imshow(this_patch)
    plt.axis("off")
    plt.title(this_class)
    
    # visualization of merged image tile patch-level prediction.
plt.show()

tile_output[0]["resolution"] = 1.0
tile_output[0]["units"] = "baseline"

label_color_dict = {}
label_color_dict[0] = ("empty", (0, 0, 0))
colors = cm.get_cmap("Set1").colors
for class_name, label in label_dict.items():
    label_color_dict[label + 1] = (class_name, 255 * np.array(colors[label]))
pred_map = predictor.merge_predictions(
    img_file_name,
    tile_output[0],
    resolution=1,
    units="baseline",
)
overlay = overlay_prediction_mask(
    input_tile,
    pred_map,
    alpha=0.5,
    label_info=label_color_dict,
    return_ax=True,
)
plt.show()

wsi_ioconfig = IOPatchPredictorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    patch_input_shape=[224, 224],
    stride_shape=[224, 224],
)

predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=64)
wsi_output = predictor.predict(
    imgs=[wsi_file_name],
    masks=None,
    mode="wsi",
    merge_predictions=False,
    ioconfig=wsi_ioconfig,
    return_probabilities=True,
    save_dir=global_save_dir / "wsi_predictions",
    device=device,
)

# visualization of whole-slide image patch-level prediction
overview_resolution = (
    4  # the resolution in which we desire to merge and visualize the patch predictions
)

# the unit of the `resolution` parameter. Can be "power", "level", "mpp", or "baseline"
overview_unit = "mpp"
wsi = WSIReader.open(wsi_file_name)
wsi_overview = wsi.slide_thumbnail(resolution=overview_resolution, units=overview_unit)
plt.figure(), plt.imshow(wsi_overview)
plt.axis("off")

pred_map = predictor.merge_predictions(
    wsi_file_name,
    wsi_output[0],
    resolution=overview_resolution,
    units=overview_unit,
)
overlay = overlay_prediction_mask(
    wsi_overview,
    pred_map,
    alpha=0.5,
    label_info=label_color_dict,
    return_ax=True,
)
plt.show()