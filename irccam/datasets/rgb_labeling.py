"""
Takes the RGB image and produces labels for each pixel using the formula Julian provided
"""
"""
Filtering of images based on various criteria
"""

import os
import cv2
import numpy as np

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data/raw/davos")
DATASET_PATH = os.path.join(PROJECT_PATH, "data/datasets")


def create_rgb_label(image, cloud_ref=2.15):
    # For the pixel cloud treshhold, this is the relevant part:
    #
    # rat=imid(:,:,3)./imid(:,:,1) + imid(:,:,3)./imid(:,:,2);
    #
    # imi_cloud = rat<cloud_ref;   % the smaller the whiter
    #
    # with cloud_ref=2.15 for this camera.
    # use ratios by Julian
    # careful RGB -> BGR

    # ugly fix for division with 0
    zeros = np.logical_or(image[:, :, 1] == 0, image[:, :, 2] == 0)
    image[:, :, 1][zeros] = 1
    image[:, :, 2][zeros] = 1

    rat = image[:, :, 0] / image[:, :, 2] + image[:, :, 0] / image[:, :, 1]
    rat[zeros] = float("inf")

    cloud = np.array(rat < cloud_ref, dtype=np.uint8)
    return cloud


def create_label_image(labels):
    img = np.zeros(labels.shape, dtype=np.uint8)
    img[np.where(labels)] = 255
    return img
