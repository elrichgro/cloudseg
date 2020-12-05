from irccam.utils.definitions import *
import cv2
import os

common_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/common_mask.bmp"), -1)
background_mask = cv2.imread(os.path.join(PROJECT_PATH, "irccam/datasets/outside_mask.bmp"), -1)
