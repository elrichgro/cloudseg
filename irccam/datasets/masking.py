from irccam.utils.constants import *
import cv2
import os
import numpy as np

full_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/resources/full_mask.bmp"), -1)
background_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/resources/background_mask.bmp"), -1)


def apply_background_mask(img):
    """
    Apply background mask to image by setting masked pixels to NaN
    """
    img[background_mask == 255] = np.nan


def apply_full_mask(img, fill=np.nan):
    """
    Apply common mask to image by setting masked pixels to the given fill value
    """
    img[full_mask == 255] = fill
