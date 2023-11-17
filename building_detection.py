import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from quicklook import quicklook

from torch.utils import model_zoo

# ignore deprecation warning from building_footprint_segmentation for now. TODO consider a PR to resolve
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

#!pip install building-footprint-segmentation
from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
    load_parallel_model,
    adjust_model,
)
from building_footprint_segmentation.utils.operations import handle_image_size


MAX_SIZE = 512
TRAINED_MODEL = ReFineNet()
MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"

# set model weights
state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
TRAINED_MODEL.load_state_dict(adjust_model(state_dict))

def predict(original_image):
    # predict building footprints from original image
    
    original_height, original_width = original_image.shape[:2]

    # handle image size
    if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
        original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))

    # Apply Normalization
    normalized_image = min_max_image_net(img=original_image)

    tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))

    # Perform prediction     
    with torch.no_grad():
        prediction = TRAINED_MODEL(tensor_image)
        prediction = prediction.sigmoid()

    binary_prediction = convert_tensor_to_numpy(prediction[0]).reshape(
        (MAX_SIZE, MAX_SIZE)
    )

    return binary_prediction

def blend_binary_into_image(original_image, binary_array):
    # "overlay" the binary prediction on to the original colour composite
    
    binary_array_as_rgb = cv2.cvtColor(binary_array, cv2.COLOR_GRAY2RGB)

    colour_prediction = cv2.addWeighted(
        original_image,
        1,
        (binary_array_as_rgb * (0, 255, 0)).astype(np.uint8),
        0.4,
        0,
    )
    return colour_prediction
    
def detect_buildings(image_path, max_size=MAX_SIZE):
    # read image from disk and create binary and colour prediction arrays
    
    arr = cv2.imread(image_path)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # model = load_parallel_model(model) # if gpu available

    # subset original image to maxsize
    subset = arr[:MAX_SIZE, :MAX_SIZE, :]
    
    binary_prediction = predict(subset)
    colour_prediction = blend_binary_into_image(subset, binary_prediction)

    return (binary_prediction, colour_prediction)

def plot_predictions(pred, title='', figsize = (12, 6)):
    # plot binary and colour prediction arrays
    
    _,ax = plt.subplots(1,2,figsize = figsize)
    ax = ax.ravel()
    for i in range(2):
      image = pred[i]
      ax[i].imshow(image)  

    plt.suptitle(title)
    plt.show()