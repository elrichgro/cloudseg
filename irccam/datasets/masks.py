from irccam.utils.constants import *
import cv2
import os

common_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/resources/common_mask.bmp"), -1)
background_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/resources/outside_mask.bmp"), -1)
