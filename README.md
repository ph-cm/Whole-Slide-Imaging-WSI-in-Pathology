# Whole-Slide-Imaging-WSI-in-Pathology

# Patch-Level Predictions Using TIAToolbox

The **TIAToolbox** is a versatile framework designed to facilitate patch-level predictions in computational pathology, supporting tasks such as image segmentation, classification, and analysis of histology images. This toolkit is particularly valuable in analyzing large image tiles or whole-slide images (WSIs), enabling efficient processing of high-resolution medical images by dividing them into smaller patches and aggregating predictions.

## Key Concepts
- **Patch-Level Predictions**: Analyze smaller image patches extracted from large histology tiles or WSIs. These patches are processed independently, and predictions are aggregated to create a comprehensive prediction map for the entire image.
- **Prediction Modes**:
  - `patch`: Processing individual pre-extracted patches.
  - `tile`: Dividing large tiles into smaller overlapping patches.
  - `wsi`: Analyzing whole-slide images by systematically extracting patches.
- **Pretrained Models**: Utilize pretrained models, such as ResNet-based architectures trained on datasets like Kather100K, enabling high accuracy without requiring training from scratch.
- **Visualization Tools**: Functions like `overlay_prediction_mask` allow visualization of prediction results as overlays on original images, making it easier to interpret and evaluate results.

## Examples Demonstrated
1. **Patch Predictions**: Process individual patches from the Kather100K dataset to predict histological tissue types such as lymphocytes, smooth muscle, and tumor epithelium.
2. **Tile Analysis**: Analyze large tiles (e.g., 5000x5000 pixels) by dividing them into smaller patches to generate prediction maps, showing the spatial distribution of various tissue types.
3. **WSI Analysis**: Process whole-slide images using `IOPatchPredictorConfig` to configure parameters like resolution, stride, and input shape.
4. **Prediction Map Merging**: Use the `merge_predictions` function to aggregate individual patch predictions and create seamless prediction maps across large regions.
5. **Visualization**: Overlay prediction maps on original WSIs with `overlay_patch_prediction` for clearer interpretation.

## Importance of the Subject
The use of **patch-level predictions in digital pathology** is a critical advancement in medical image analysis. It offers:
1. **Efficient Processing**: Break large and complex images into manageable units, enabling high-throughput analysis.
2. **Improved Accuracy**: Reduce errors and enhance the precision of histological classifications using pretrained models and systematic patch-based processing.
3. **Visualization and Interpretation**: Create prediction maps to better understand spatial relationships and tissue heterogeneity within a slide.
4. **Versatility**: Customize the framework for different datasets, model architectures, and specific clinical or research applications.
5. **Accessibility**: Built-in tools for configuration, training, and visualization make the framework usable even for users with minimal expertise in machine learning.

## Conclusion
TIAToolbox represents a significant step forward in automating and scaling digital pathology workflows. By providing tools to perform patch-level predictions, aggregate results into meaningful maps, and visualize outcomes, it bridges the gap between computational pathology and clinical applications. These capabilities are instrumental in advancing cancer diagnostics, treatment planning, and biomedical research.


#About the application

In this example, we will show how to use TIAToolbox for patch-level prediction using a range of deep learning models. TIAToolbox can be used to make predictions on pre-extracted image patches or on larger image tiles / whole-slide images (WSIs), where image patches are extracted on the fly. WSI patch-level predictions can subsequently be aggregated to obtain a segmentation map. In particular, we will introduce the use of our module `patch_predictor`. A full list of the available models trained and provided in TIAToolbox for patch-level prediction is given below.

# Importing related libraries

We import some standard Python modules, and also the TIAToolbox Python modules for the patch classification task, written by the TIA Centre team.

# Downloading the required files

We download, over the internet, image files used for the purpose of this notebook. In particular, we download a sample subset of validation patches that were used when training models on the Kather 100k dataset, a sample image tile, and a sample whole-slide image. Downloading is needed once in each Colab session and it should take less than 1 minute. In Colab, if you click the file's icon (see below) in the vertical toolbar on the left-hand side, then you can see all the files which the code in this notebook can access. The data will appear here when it is downloaded.

# Get predictions for a set of patches

Below we use `tiatoolbox` to obtain the model predictions for a set of patches with a pretrained model.

We use patches from the validation subset of the Kather 100k dataset. This dataset has already been downloaded in the download section above. We first read the data and convert it to a suitable format. In particular, we create a list of patches and a list of corresponding labels. For example, the first label in `label_list` will indicate the class of the first image patch in `patch_list`.

# Patch Dataset Classes and Labels

As you can see for this patch dataset, we have 9 classes/labels with IDs 0-8 and associated class names, describing the dominant tissue type in the patch:

- **BACK** → Background (empty glass region)
- **LYM** → Lymphocytes
- **NORM** → Normal colon mucosa
- **DEB** → Debris
- **MUS** → Smooth muscle
- **STR** → Cancer-associated stroma
- **ADI** → Adipose
- **MUC** → Mucus
- **TUM** → Colorectal adenocarcinoma epithelium

It is easy to use this code for your dataset – just ensure that your dataset is arranged like this example (images of different classes are placed into different subfolders), and set the right image extension in the `image_ext` variable.

# Predict Patch Labels in 2 Lines of Code

Now that we have the list of images, we can use TIAToolbox's `PatchPredictor` to predict their category. First, we instantiate a predictor object and then we call the `predict` method to get the results.

# Patch Prediction is Done!

The first line creates a CNN-based patch classifier instance based on the arguments and prepares a CNN model (generates the network, downloads pretrained weights, etc.). The CNN model used in this predictor can be defined using the `pretrained_model` argument. A complete list of supported pretrained classification models, that have been trained on the Kather 100K dataset, is reported in the first section of this notebook. `PatchPredictor` also enables you to use your own pretrained models for your specific classification application. In order to do that, you might need to change some input arguments for `PatchPredictor`, as we now explain:

- **model**: Use an externally defined PyTorch model for prediction, with weights already loaded. This is useful when you want to use your own pretrained model on your own data. The only constraint is that the input model should follow `tiatoolbox.models.abc.ModelABC` class structure. For more information on this matter, please refer to our example notebook on advanced model techniques.
- **pretrained_model**: This argument has already been discussed above. With it, you can tell tiatoolbox to use one of its pretrained models for the prediction task. If both `model` and `pretrained_model` arguments are used, then `pretrained_model` is ignored. In this example, we used `resnet18-kather100k`, which means that the model architecture is an 18-layer ResNet, trained on the Kather100k dataset.
- **pretrained_weight**: When using a `pretrained_model`, the corresponding pretrained weights will also be downloaded by default. You can override the default with your own set of weights via the `pretrained_weight` argument.
- **batch_size**: Number of images fed into the model each time. Higher values for this parameter require a larger (GPU) memory capacity.
- 
# Predict Method Input Arguments

The second line in the snippet above calls the `predict` method to apply the CNN on the input patches and get the results. Here are some important `predict` input arguments and their descriptions:

- **mode**: Type of input to be processed. Choose from `patch`, `tile`, or `wsi` according to your application. In this first example, we predict the tissue type of histology patches, so we use the `patch` option. The use of `tile` and `wsi` options are explained below.
- **imgs**: List of inputs. When using `patch` mode, the input must be a list of images OR a list of image file paths, OR a Numpy array corresponding to an image list. However, for the `tile` and `wsi` modes, the `imgs` argument should be a list of paths to the input tiles or WSIs.
- **return_probabilities**: Set to `True` to get per-class probabilities alongside predicted labels of input patches. If you wish to merge the predictions to generate prediction maps for `tile` or `wsi` modes, you can set `return_probabilities=True`.

In the `patch` prediction mode, the `predict` method returns an output dictionary that contains the `predictions` (predicted labels) and `probabilities` (probability that a certain patch belongs to a certain class).

The cell below uses common Python tools to visualize the patch classification results in terms of classification accuracy and confusion matrix.

# Get Predictions for Patches Within an Image Tile

We now demonstrate how to obtain patch-level predictions for a large image tile. It is quite a common practice in computational pathology to divide a large image into several patches (often overlapping) and then aggregate the results to generate a prediction map for different regions of the large image. As we are making a prediction per patch again, there is no need to instantiate a new `PatchPredictor` class. However, we should tune the `predict` input arguments to make them suitable for tile prediction. The `predict` function then automatically extracts patches from the large image tile and predicts the label for each of them. 

As the `predict` function can accept multiple tiles in the input to be processed and each input tile has potentially many patches, we save results in a file when more than one image is provided. This is done to avoid any problems with limited computer memory. However, if only one image is provided, the results will be returned as in `patch` mode.

Now, we try this function on a sample image tile. For this example, we use a tile that was released with the Kather et al. 2016 paper. It has been already downloaded in the Download section of this notebook.

# Patch-Level Prediction in 2 Lines of Code for Big Histology Tiles

As you can see, the size of the tile image is 5000x5000 pixels. This is quite big and might result in computer memory problems if fed directly into a deep learning model. However, the `predict` method of `PatchPredictor` handles this big tile seamlessly by processing small patches independently. You only need to change the `mode` argument to `tile` and a couple of other arguments, as explained below.

The new arguments in the input of the `predict` method are:

- **mode='tile'**: The type of image input. We use `tile` since our input is a large image tile.
- **imgs**: In tile mode, the input is required to be a list of file paths.
- **save_dir**: Output directory when processing multiple tiles. We explained before why this is necessary when we are working with multiple big tiles.
- **patch_size**: This parameter sets the size of patches (in [W, H] format) to be extracted from the input files, and for which labels will be predicted.
- **stride_size**: The stride (in [W, H] format) to consider when extracting patches from the tile. Using a stride smaller than the patch size results in overlapping between consecutive patches.
- **labels** (optional): List of labels with the same size as `imgs` that refers to the label of each input tile (not to be confused with the prediction of each patch).

In this example, we input only one tile. Therefore, the toolbox does not save the output as files and instead returns a list that contains an output dictionary with the following keys:

- **coordinates**: List of coordinates of extracted patches in the following format: `[x_min, y_min, x_max, y_max]`. These coordinates can be used to later extract the same region from the input tile/WSI or regenerate a prediction map based on the `prediction` labels for each patch.
- **predictions**: List of predicted labels for each of the tile’s patches.
- **label**: Label of the tile generalized to each patch.

---

Keep in mind that if we had several items in the `imgs` input, then the result would be saved in JSON format to the specified `save_dir` and the returned output will be a list of paths to each of the saved JSON files.

---

# Visualisation of Tile Results

Below we will show some of the results generated by our patch-level predictor on the input image tile. First, we will show some individual patch predictions and then we will show the merged patch-level results on the entire image tile.

# Generating and Visualizing the Prediction Map

Here, we show a prediction map where each color denotes a different predicted category. We overlay the prediction map on the original image. To generate this prediction map, we utilize the `merge_predictions` method from the `PatchPredictor` class which accepts as arguments the path of the original image, `predictor` outputs, `mode` (set to `tile` or `wsi`), `tile_resolution` (at which tiles were originally extracted), and `resolution` (at which the prediction map is generated), and outputs the "Prediction map," in which regions have indexed values based on their classes.

To visualize the prediction map as an overlay on the input image, we use the `overlay_prediction_mask` function from the `tiatoolbox.utils.visualization` module. It accepts as arguments the original image, the prediction map, the `alpha` parameter which specifies the blending ratio of overlay and original image, and the `label_info` dictionary which contains names and desired colors for different classes. Below we generate an example of an acceptable `label_info` dictionary and show how it can be used with `overlay_patch_prediction`.

# Get Predictions for Patches Within a WSI

We demonstrate how to obtain predictions for all patches within a whole-slide image. As in previous sections, we will use `PatchPredictor` and its `predict` method, but this time we set the `mode` to `'wsi'`. 

We also introduce `IOPatchPredictorConfig`, a class that specifies the configuration of image reading and prediction writing for the model prediction engine.
# Parameters of `IOPatchPredictorConfig`

Parameters of `IOPatchPredictorConfig` have self-explanatory names, but let’s have a look at their definitions:

- **input_resolutions**: A list specifying the resolution of each input head of the model in the form of a dictionary. List elements must be in the same order as the target `model.forward()`. Of course, if your model accepts only one input, you just need to put one dictionary specifying `'units'` and `'resolution'`. But it’s good to know that TIAToolbox supports a model with more than one input.
- **patch_input_shape**: Shape of the largest input in (height, width) format.
- **stride_shape**: The size of stride (steps) between two consecutive patches, used in the patch extraction process. If the user sets `stride_shape` equal to `patch_input_shape`, patches will be extracted and processed without any overlap.

Now that we have set everything, we try our patch predictor on a WSI. Here, we use a large WSI and therefore the patch extraction and prediction processes may take some time (make sure to set the `device="cuda"` if you have access to a Cuda-enabled GPU and PyTorch+Cuda).

# New Arguments for `predict` Method

We introduce some new arguments for the `predict` method:

- **mode**: Set to `wsi` when analyzing whole-slide images.
- **ioconfig**: Set the IO configuration information using the `IOPatchPredictorConfig` class.
- **resolution** and **unit** (not shown above): These arguments specify the level or micron-per-pixel resolution of the WSI levels from which we plan to extract patches and can be used instead of `ioconfig`. Here we specify the WSI's level as `'baseline'`, which is equivalent to level 0. In general, this is the level of greatest resolution. In this particular case, the image has only one level. More information can be found in the documentation.
- **masks**: A list of paths corresponding to the masks of WSIs in the `imgs` list. These masks specify the regions in the original WSIs from which we want to extract patches. If the mask of a particular WSI is specified as `None`, then the labels for all patches of that WSI (even background regions) would be predicted. This could cause unnecessary computation.
- **merge_predictions**: You can set this parameter to `True` if you wish to generate a 2D map of patch classification results. However, for big WSIs, you might need a large amount of memory available to do this on the file. An alternative (default) solution is to set `merge_predictions=False`, and then generate the 2D prediction maps using the `merge_predictions` function as you will see later on.

# Visualizing Whole-Slide Image (WSI) Output

We see how the prediction model works on our whole-slide images by visualizing the `wsi_output`. We first need to merge patch prediction outputs and then visualize them as an overlay on the original image. As before, the `merge_predictions` method is used to merge the patch predictions. 

Here we set the parameters `resolution=1.25`, `units='power'` to generate the prediction map at 1.25x magnification. If you would like to have higher/lower resolution (bigger/smaller) prediction maps, you need to change these parameters accordingly. 

When the predictions are merged, use the `overlay_patch_prediction` function to overlay the prediction map on the WSI thumbnail, which should be extracted at the same resolution used for prediction merging.
# Summary of Patch Prediction with TIAToolbox

In this notebook, we show how we can use the `PatchPredictor` class and its `predict` method to predict the label for patches of big tiles and WSIs. We introduce `merge_predictions` and `overlay_prediction_mask` helper functions that merge the patch prediction outputs and visualize the resulting prediction map as an overlay on the input image/WSI.

All the processes take place within TIAToolbox, and you can easily put the pieces together by following our example code. Just make sure to set inputs and options correctly. We encourage you to further investigate the effect on the prediction output of changing `predict` function parameters. 

Furthermore, if you want to use your own pretrained model for patch classification in the TIAToolbox framework (even if the model structure is not defined in the TIAToolbox model class), you can follow the instructions in our example notebook on advanced model techniques to gain some insights and guidance.





